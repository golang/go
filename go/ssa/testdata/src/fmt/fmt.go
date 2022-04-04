package fmt

func Sprint(args ...interface{}) string
func Sprintln(args ...interface{}) string
func Sprintf(format string, args ...interface{}) string

func Print(args ...interface{}) (int, error)
func Println(args ...interface{})
func Printf(format string, args ...interface{}) (int, error)

func Errorf(format string, args ...interface{}) error
