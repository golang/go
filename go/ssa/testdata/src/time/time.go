package time

type Duration int64

func Sleep(Duration)

func NewTimer(d Duration) *Timer

type Timer struct {
	C <-chan Time
}

func (t *Timer) Stop() bool

type Time struct{}

func After(d Duration) <-chan Time

const (
	Nanosecond Duration = iota // Specific values do not matter here.
	Second
	Minute
	Hour
)
