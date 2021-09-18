package log

import (
	"context"
	"fmt"
	"os"
)

// The Entry is a logging entry that contains context set by the user and needed data
// for the log.
// Entry fields are used by a custom formatter while formatting output.
type Entry struct {
	logger    *Logger         // logger which will be used to log the entry
	context   context.Context // context set by the user
	calldepth int             // calldepth is the count of the number of frames to skip
	level     *Level          // level of the entry
	message   string          // message contains the text to print
}

// NewEntry creates a new Entry. The logger variable sets the
// the logger which will be used to log the entry.
func NewEntry(logger *Logger) *Entry {
	return &Entry{
		logger: logger,
	}
}

// Logger returns the logger which will write entry to the output destination.
func (e *Entry) Logger() *Logger {
	return e.logger
}

// Context returns the context set by the user for entry.
func (e *Entry) Context() context.Context {
	return e.context
}

// LogLevel returns the log level for entry.
func (e *Entry) LogLevel() *Level {
	return e.level
}

// Message returns the log message for entry.
func (e *Entry) Message() string {
	return e.message
}

// CallDepth returns the calldepth for entry.
func (e *Entry) CallDepth() int {
	return e.calldepth
}

// Print calls e.Output to print to the logger.
// Arguments are handled in the manner of fmt.Print.
func (e *Entry) Print(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...))
}

// Printf calls e.Output to print to the logger.
// Arguments are handled in the manner of fmt.Printf.
func (e *Entry) Printf(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...))
}

// Println calls e.Output to print to the logger.
// Arguments are handled in the manner of fmt.Println.
func (e *Entry) Println(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...))
}

// Fatal is equivalent to Print() followed by a call to os.Exit(1).
func (e *Entry) Fatal(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), FatalLevel)
	os.Exit(1)
}

// Fatalf is equivalent to Printf() followed by a call to os.Exit(1).
func (e *Entry) Fatalf(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), FatalLevel)
	os.Exit(1)
}

// Fatalln is equivalent to Println() followed by a call to os.Exit(1).
func (e *Entry) Fatalln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), FatalLevel)
	os.Exit(1)
}

// Panic is equivalent to Print() and logs the message at level Error
// followed by a call to panic().
func (e *Entry) Panic(v ...interface{}) {
	s := fmt.Sprint(v...)
	e.Output(2, s, PanicLevel)
	panic(s)
}

// Panicf is equivalent to Printf() and logs the message at level Error
// followed by a call to panic().
func (e *Entry) Panicf(format string, v ...interface{}) {
	s := fmt.Sprintf(format, v...)
	e.Output(2, s, PanicLevel)
	panic(s)
}

// Panicln is equivalent to Println() and logs the message at level Error
// followed by a call to panic().
func (e *Entry) Panicln(v ...interface{}) {
	s := fmt.Sprintln(v...)
	e.Output(2, s, PanicLevel)
	panic(s)
}

// Error is equivalent to Print() and logs the message at level Error.
func (e *Entry) Error(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), ErrorLevel)
}

// Errorf is equivalent to Printf() and logs the message at level Error.
func (e *Entry) Errorf(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), ErrorLevel)
}

// Errorln is equivalent to Println() and logs the message at level Error.
func (e *Entry) Errorln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), ErrorLevel)
}

// Warn is equivalent to Print() and logs the message at level Warning.
func (e *Entry) Warn(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), WarnLevel)
}

// Warnf is equivalent to Printf() and logs the message at level Warning.
func (e *Entry) Warnf(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), WarnLevel)
}

// Warnln is equivalent to Println() and logs the message at level Warning.
func (e *Entry) Warnln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), WarnLevel)
}

// Info is equivalent to Print() and logs the message at level Info.
func (e *Entry) Info(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), InfoLevel)
}

// Infof is equivalent to Printf() and logs the message at level Info.
func (e *Entry) Infof(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), InfoLevel)
}

// Infoln is equivalent to Println() and logs the message at level Info.
func (e *Entry) Infoln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), InfoLevel)
}

// Debug is equivalent to Print() and logs the message at level Debug.
func (e *Entry) Debug(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), DebugLevel)
}

// Debugf is equivalent to Printf() and logs the message at level Debug.
func (e *Entry) Debugf(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), DebugLevel)
}

// Debugln is equivalent to Println() and logs the message at level Debug.
func (e *Entry) Debugln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), DebugLevel)
}

// Trace is equivalent to Print() and logs the message at level Trace.
func (e *Entry) Trace(v ...interface{}) {
	e.Output(2, fmt.Sprint(v...), TraceLevel)
}

// Tracef is equivalent to Printf() and logs the message at level Trace.
func (e *Entry) Tracef(format string, v ...interface{}) {
	e.Output(2, fmt.Sprintf(format, v...), TraceLevel)
}

// Traceln is equivalent to Println() and logs the message at level Trace.
func (e *Entry) Traceln(v ...interface{}) {
	e.Output(2, fmt.Sprintln(v...), TraceLevel)
}

// Output writes the output for a logging event. The string s contains
// the text to print after the prefix specified by the flags of the
// Logger. A newline is appended if the last character of s is not
// already a newline. Calldepth is the count of the number of
// frames to skip when computing the file name and line number
// if Llongfile or Lshortfile is set; a value of 1 will print the details
// for the caller of Output. Level is the log level for the output.
// If any formatter is configured for the logger, it will be used to format
// the output.
func (e *Entry) Output(calldepth int, s string, level ...Level) error {
	var formatter LoggerFormatter

	e.logger.mu.Lock()
	if e.logger.rootLogger != nil {
		e.logger.rootLogger.formatterMu.Lock()
		formatter = e.logger.rootLogger.formatter
		e.logger.rootLogger.formatterMu.Unlock()
	}
	if formatter == nil {
		formatter = e.logger.formatter
	}
	e.logger.mu.Unlock()

	if formatter != nil {
		// +1 for this frame.
		e.calldepth = calldepth + 1
		e.message = s

		if level != nil {
			e.level = &level[0]
		} else {
			e.level = nil
		}

		serialized, err := formatter.Format(e)

		if err == nil && serialized != nil {
			// if the logger has got a root logger, use the output
			// destination of the root logger.
			if e.logger.rootLogger != nil {
				e.logger.rootLogger.mu.Lock()
				_, err = e.logger.rootLogger.out.Write(serialized)
				e.logger.rootLogger.mu.Unlock()
			} else {
				e.logger.mu.Lock()
				_, err = e.logger.out.Write(serialized)
				e.logger.mu.Unlock()
			}
		}

		return err
	}

	return e.logger.Output(calldepth+1, s, level...) // +1 for this frame.
}
