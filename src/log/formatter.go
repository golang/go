package log

// The LoggerFormatter interface is used to implement a custom formatter.
// So the log output can be customized by implementing this interface.
type LoggerFormatter interface {
	Format(entry *Entry) ([]byte, error)
}
