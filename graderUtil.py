"""
Library to do grading of Python programs.
Usage (see grader.py):

    # create a grader
    grader = Grader("Name of assignment")

    # add a basic test
    grader.addBasicPart(number, gradeFunc, maxPoints, maxSeconds, description="a basic test")

    # add a hidden test
    grader.addHiddenPart(number, gradeFunc, maxPoints, maxSeconds, description="a hidden test")

    # add a manual grading part
    grader.addManualPart(number, gradeFunc, maxPoints, description="written problem")

    # run grading
    grader.grade()
"""


import argparse
import ctypes
import threading
import datetime, math, pprint, traceback, sys, signal, os, json
import gc

defaultMaxSeconds = None  # 5 seconds or None for no timeout
TOLERANCE = 1e-4  # For measuring whether two floats are equal

BASIC_MODE = 'basic'  # basic
AUTO_MODE = 'auto'    # basic + hidden
ALL_MODE = 'all'      # basic + hidden + manual

# When reporting stack traces as feedback, ignore parts specific to the grading
# system.
def isTracebackItemGrader(item):
    return item[0].endswith('graderUtil.py')

def isCollection(x):
    return isinstance(x, list) or isinstance(x, tuple)

# Return whether two answers are equal.
def isEqual(trueAnswer, predAnswer, tolerance = TOLERANCE):
    # Handle floats specially
    if isinstance(trueAnswer, float) or isinstance(predAnswer, float):
        return abs(trueAnswer - predAnswer) < tolerance
    # Recurse on collections to deal with floats inside them
    if isCollection(trueAnswer) and isCollection(predAnswer) and len(trueAnswer) == len(predAnswer):
        for a, b in zip(trueAnswer, predAnswer):
            if not isEqual(a, b): return False
        return True
    if isinstance(trueAnswer, dict) and isinstance(predAnswer, dict):
        if len(trueAnswer) != len(predAnswer): return False
        for k, v in list(trueAnswer.items()):
            if not isEqual(predAnswer.get(k), v): return False
        return True

    # Numpy array comparison
    if type(trueAnswer).__name__ == 'ndarray':
        import numpy as np
        if isinstance(trueAnswer, np.ndarray) and isinstance(predAnswer, np.ndarray):
            if trueAnswer.shape != predAnswer.shape:
                return False
            for a, b in zip(trueAnswer, predAnswer):
                if not isEqual(a, b): return False
            return True

    # Do normal comparison
    return trueAnswer == predAnswer

# Run a function, timing out after maxSeconds.
class TimeoutFunctionException(Exception):
    pass

class TimerManager:
    def __init__(self):
        self.timer = None

    def set_alarm(self, seconds, handler):
        if self.timer:
            self.timer.cancel()  # Cancel any existing timer
        self.timer = threading.Timer(seconds, handler)
        self.timer.start()

    def cancel_alarm(self):
        if self.timer:
            self.timer.cancel()
            self.timer = None

class TimeoutFunction:
    def __init__(self, function, maxSeconds):
        self.function = function
        self.maxSeconds = maxSeconds
        self.timer_manager = TimerManager()

    def handle_maxSeconds(self):
        raise TimeoutFunctionException("Function execution exceeded the timeout.")

    def __call__(self, *args):
        # No timeout
        if self.maxSeconds is None:
            return self.function(*args)
        
        if os.name == 'nt':
            # Windows version using TimerManager
            timeStart = datetime.datetime.now()
            self.timer_manager.set_alarm(self.maxSeconds + 1, self.handle_maxSeconds)
            try:
                result = self.function(*args)
            finally:
                self.timer_manager.cancel_alarm()  # Cancel the timer regardless of outcome
            timeEnd = datetime.datetime.now()
            if timeEnd - timeStart > datetime.timedelta(seconds=self.maxSeconds + 1):
                raise TimeoutFunctionException("Function execution exceeded the timeout.")
            return result
            
        else:               
            # Unix implementation using signal
            import signal

            def handle_timeout(signum, frame):
                raise TimeoutFunctionException("Function execution exceeded the timeout.")

            old_handler = signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(self.maxSeconds + 1)

            try:
                self.result = self.function(*args)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return self.result
class Part:
    def __init__(self, number, gradeFunc, maxPoints, maxSeconds, extraCredit, description, basic):
        if not isinstance(number, str):
            raise Exception("Invalid number: %s" % number)
        if gradeFunc != None and not callable(gradeFunc):
            raise Exception("Invalid gradeFunc: %s" % gradeFunc)
        if not isinstance(maxPoints, int):
            raise Exception("Invalid maxPoints: %s" % maxPoints)
        if maxSeconds != None and not isinstance(maxSeconds, int):
            raise Exception("Invalid maxSeconds: %s" % maxSeconds)
        if not description:
            print('ERROR: description required for part {}'.format(number))
        # Specification of part
        self.number = number  # Unique identifier for this part.
        self.description = description  # Description of this part
        self.gradeFunc = gradeFunc  # Function to call to do grading
        self.maxPoints = maxPoints  # Maximum number of points attainable on this part
        self.maxSeconds = maxSeconds  # Maximum allowed time that the student's code can take (in seconds)
        self.extraCredit = extraCredit  # Whether this is an extra credit problem
        self.basic = basic
        # Grading the part
        self.points = 0
        self.side = None  # Side information
        self.seconds = 0
        self.messages = []
        self.failed = False

    def fail(self):
        self.failed = True

    def is_basic(self):
        return self.gradeFunc is not None and self.basic
    def is_hidden(self):
        return self.gradeFunc is not None and not self.basic
    def is_auto(self):
        return self.gradeFunc is not None
    def is_manual(self):
        return self.gradeFunc is None

class Grader:
    def __init__(self, args=sys.argv):
        self.parts = []  # Parts (to be added)
        self.useSolution = False  # Set to true if we are actually evaluating the hidden test cases

        parser = argparse.ArgumentParser()
        parser.add_argument('--js', action='store_true', help='Write JS file with information about this assignment')
        parser.add_argument('--json', action='store_true', help='Write JSON file with information about this assignment')
        parser.add_argument('--summary', action='store_true', help='Don\'t actually run code')
        parser.add_argument('-q', type=str, metavar='qn', help='Test only the specified question')
        parser.add_argument('--no-graphics', action='store_true', help='Disable graphics by overriding plt.show()')
        parser.add_argument('--all', action='store_true', help='Grade all questions')

        # parser.add_argument('remainder', nargs=argparse.REMAINDER)
        self.params = parser.parse_args(args[1:])

        self.mode = ALL_MODE

        # args = self.params.remainder
        # if len(args) < 1:
        #     self.mode = AUTO_MODE
        #     self.selectedPartName = None
        # else:
        #     if args[0] in [BASIC_MODE, AUTO_MODE, ALL_MODE]:
        #         self.mode = args[0]
        #         self.selectedPartName = None
        #     else:
        #         self.mode = AUTO_MODE
        #         self.selectedPartName = args[0]

        self.messages = []  # General messages
        self.currentPart = None  # Which part we're grading
        self.fatalError = False  # Set this if we should just stop immediately
        cwd = os.getcwd()
        assignment_name = cwd.split('/')[-1]
        num_points = 1
        if 'p-' in assignment_name:
            num_points = 0
        # self.addManualPart('style', maxPoints=num_points, extraCredit=True, description='whether writeup is nicely typed, etc.')

        if self.params.no_graphics:
            self.handleNoGraphics()

    def addBasicPart(self, number, gradeFunc, maxPoints=1, maxSeconds=defaultMaxSeconds, extraCredit=False, description=""):
        """Add a basic test case. The test will be visible to students."""
        self.assertNewNumber(number)
        part = Part(number, gradeFunc, maxPoints, maxSeconds, extraCredit, description, basic=True)
        self.parts.append(part)

    def addHiddenPart(self, number, gradeFunc, maxPoints=1, maxSeconds=defaultMaxSeconds, extraCredit=False, description=""):
        """Add a hidden test case. The output should NOT be visible to students and so should be inside a BEGIN_HIDE block."""
        self.assertNewNumber(number)
        part = Part(number, gradeFunc, maxPoints, maxSeconds, extraCredit, description, basic=False)
        self.parts.append(part)

    def addManualPart(self, number, maxPoints, extraCredit=False, description=""):
        """Add a manual part."""
        self.assertNewNumber(number)
        part = Part(number, None, maxPoints, None, extraCredit, description, basic=False)
        self.parts.append(part)

    def assertNewNumber(self, number):
        if number in [part.number for part in self.parts]:
            raise Exception("Part number %s already exists" % number)

    # Try to load the module (submission from student).
    def load(self, moduleName):
        try:
            return __import__(moduleName)
        except Exception as e:
            self.fail("Threw exception when importing '%s': %s" % (moduleName, e))
            self.fatalError = True
            return None
        except:
            self.fail("Threw exception when importing '%s'" % moduleName)
            self.fatalError = True
            return None

    def gradePart(self, part):
        print('----- START PART %s%s: %s' % (part.number, ' (extra credit)' if part.extraCredit else '', part.description))
        self.currentPart = part

        startTime = datetime.datetime.now()
        try:
            TimeoutFunction(part.gradeFunc, part.maxSeconds)()  # Call the part's function
        except KeyboardInterrupt:
            raise
        except MemoryError as e:
            if os.name != 'nt':
                signal.alarm(0)
            gc.collect()
            self.fail('Memory limit exceeded.')
        except TimeoutFunctionException as e:
            if os.name != 'nt':
                signal.alarm(0)
            self.fail('Time limit (%s seconds) exceeded.' % part.maxSeconds)
        except Exception as e:
            if os.name != 'nt':
                signal.alarm(0)
            self.fail('Exception thrown: %s -- %s' % (str(type(e)), str(e)))
            self.printException()
        except SystemExit as e:
            # Catch SystemExit raised by exit(), quit() or sys.exit()
            # This class is not a subclass of Exception and we don't
            # expect students to raise it.
            self.fail('Unexpected exit.')
            self.printException()
        endTime = datetime.datetime.now()
        part.seconds = (endTime - startTime).seconds
        if part.is_hidden() and not self.useSolution:
            displayPoints = '???/%s points (hidden test ungraded)' % part.maxPoints
        else:
            displayPoints = '%s/%s points' % (part.points, part.maxPoints)
        if part.maxSeconds:
            print('----- END PART %s [took %s (max allowed %s seconds), %s]' % (part.number, endTime - startTime, part.maxSeconds, displayPoints))
        else:
            print('----- END PART %s [took %s, %s]' % (part.number, endTime - startTime, displayPoints))
        print()

    def getSelectedParts(self):
        parts = []
        for part in self.parts:
            if self.params.q and self.params.q != part.number:
                continue
            if self.params.all or self.params.q:
                parts.append(part)
            elif self.params.summary:
                # If summary mode, include all parts for display purposes
                parts.append(part)
            elif self.mode == BASIC_MODE and part.is_basic():
                parts.append(part)
            elif self.mode == AUTO_MODE and part.is_auto():
                parts.append(part)
            elif self.mode == ALL_MODE:
                parts.append(part)
            else:
                raise Exception("Invalid mode: {}".format(self.mode))
        return parts

    def grade(self):
        parts = self.getSelectedParts()

        result = {}
        result['mode'] = self.mode

        # Grade it!
        if not self.params.summary and not self.fatalError:
            print('========== START GRADING')
            for part in parts:
                self.gradePart(part)

            # When students have it (not useSolution), only include basic tests.
            activeParts = [part for part in parts if self.useSolution or part.basic]

            totalPoints = sum(part.points for part in activeParts if not part.extraCredit)
            extraCredit = sum(part.points for part in activeParts if part.extraCredit)
            maxTotalPoints = sum(part.maxPoints for part in activeParts if not part.extraCredit)
            maxExtraCredit = sum(part.maxPoints for part in activeParts if part.extraCredit)

            if not self.useSolution:
                print('Note that the hidden test cases do not check for correctness.' \
                '\nThey are provided for you to verify that the functions do not crash and run within the time limit.' \
                '\nPoints for these parts not assigned by the grader (indicated by "--").')
            print('========== END GRADING [%d/%d points + %d/%d extra credit]' % \
                (totalPoints, maxTotalPoints, extraCredit, maxExtraCredit))

        resultParts = []
        leaderboard = []
        for part in parts:
            r = {}
            r['number'] = part.number
            r['name'] = part.description

            if self.params.summary:
                # Just print out specification of the part
                r['description'] = part.description
                r['maxSeconds'] = part.maxSeconds
                r['maxPoints'] = part.maxPoints
                r['extraCredit'] = part.extraCredit
                r['basic'] = part.basic
            else:
                r['score'] = part.points
                r['max_score'] = part.maxPoints
                r["visibility"] = "after_published" if part.is_hidden() else "visible"
                r['seconds'] = part.seconds
                if part.side is not None:
                    r['side'] = part.side
                r['output'] = "\n".join(part.messages)

                if part.side is not None:
                    for k in part.side:
                        leaderboard.append({"name" : k, "value" : part.side[k]})
            resultParts.append(r)
        result['tests'] = resultParts
        result['leaderboard'] = leaderboard

        self.output(self.mode, result)

        def display(name, extraCredit):
            parts = [part for part in self.parts if part.extraCredit == extraCredit]
            maxBasicPoints = sum(part.maxPoints for part in parts if part.is_basic())
            maxHiddenPoints = sum(part.maxPoints for part in parts if part.is_hidden())
            maxManualPoints = sum(part.maxPoints for part in parts if part.is_manual())
            maxTotalPoints = maxBasicPoints + maxHiddenPoints + maxManualPoints
            print("Total %s (basic auto/coding + hidden auto/coding + manual/written): %d + %d + %d = %d" % \
                (name, maxBasicPoints, maxHiddenPoints, maxManualPoints, maxTotalPoints))
            if not extraCredit and maxTotalPoints != 75:
                print('WARNING: maxTotalPoints = {} is not 75'.format(maxTotalPoints))
        if self.params.summary:
            display('points', False)
            display('extra credit', True)

    def output(self, mode, result):
        if self.params.json:
            path = 'grader-{}.json'.format(mode)
            with open(path, 'w') as out:
                print(json.dumps(result), file=out)
            print('Wrote to %s' % path)
        if self.params.js:
            path = 'grader-{}.js'.format(mode)
            with open(path, 'w') as out:
                print('var ' + mode + 'Result = '+ json.dumps(result) + ';', file=out)
            print('Wrote to %s' % path)

    def handleNoGraphics(self):
        if "matplolib" not in sys.modules:
            import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            import matplotlib.pyplot as plt


        # Use the Agg backend to avoid opening a window
        matplotlib_module = sys.modules["matplotlib"]
        matplotlib_module.use('Agg')

        # Retrieve the module reference
        plt = sys.modules["matplotlib.pyplot"]

        def plt_show_no_op(*args, **kwargs):
            print("plt.show() has been overridden and will not display a window.")

        plt.show = plt_show_no_op

    # Called by the grader to modify state of the current part

    def addPoints(self, amt):
        self.currentPart.points += amt

    def assignFullCredit(self):
        if not self.currentPart.failed:
            self.currentPart.points = self.currentPart.maxPoints
        return True

    def assignPartialCredit(self, credit):
        self.currentPart.points = credit
        return True;

    def setSide(self, side):
        self.currentPart.side = side

    def truncateString(self, string, length=200):
        if len(string) <= length:
            return string
        else:
            return string[:length] + '...'

    def requireIsNumeric(self, answer):
        if isinstance(answer, int) or isinstance(answer, float):
            return self.assignFullCredit()
        else:
            return self.fail("Expected either int or float, but got '%s'" % self.truncateString(answer))

    def requireIsOneOf(self, trueAnswers, predAnswer):
        if predAnswer in trueAnswers:
            return self.assignFullCredit()
        else:
            return self.fail("Expected one of %s, but got '%s'" % (self.truncateString(trueAnswers), self.truncateString(predAnswer)))

    def requireIsEqual(self, trueAnswer, predAnswer, tolerance = TOLERANCE):
        if isEqual(trueAnswer, predAnswer, tolerance):
            return self.assignFullCredit()
        else:
            return self.fail("Expected '%s', but got '%s'" % (self.truncateString(str(trueAnswer)), self.truncateString(str(predAnswer))))

    def requireIsLessThan(self, lessThanQuantity, predAnswer ):
        if predAnswer < lessThanQuantity:
            return self.assignFullCredit()
        else:
            return self.fail("Expected to be < %f, but got %f" % (lessThanQuantity, predAnswer) )

    def requireIsGreaterThan(self, greaterThanQuantity, predAnswer ):
        if predAnswer > greaterThanQuantity:
            return self.assignFullCredit()
        else:
            return self.fail("Expected to be > %f, but got %f" %
                    (greaterThanQuantity, predAnswer) )

    def requireIsTrue(self, predAnswer):
        if predAnswer:
            return self.assignFullCredit()
        else:
            return self.fail("Expected to be true, but got false" )

    def fail(self, message):
        # print('FAIL:', message)
        # self.addMessage(message)
        print('FAIL: ', end='')
        if self.currentPart:
            self.currentPart.points = 0
            self.currentPart.fail()
        return False

    def printException(self):
        tb = [item for item in traceback.extract_tb(sys.exc_info()[2]) if not isTracebackItemGrader(item)]
        for item in traceback.format_list(tb):
            self.fail('%s' % item)

    def addMessage(self, message):
        if not self.useSolution:
            print(message)
        if self.currentPart:
            self.currentPart.messages.append(message)
        else:
            self.messages.append(message)
