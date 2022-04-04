package json

func Marshal(v any) ([]byte, error)
func Unmarshal(data []byte, v any) error
