package enumtest

import "cmd/vet/testdata/enum/model"

func load() (model.Decision, error) { return model.Decision.Allow{}, nil }

func save() error { return nil }

func inspect(decision model.Decision) (string, error) {
	loaded, err := load()
	if err != nil {
		return "", err
	}
	if err := save(); err != nil {
		return "", err
	}
	qualified := model.Decision.Deny{Reason: "no"}
	_ = qualified
	switch loaded {
	case Allow:
		return "yes", nil
	case Deny:
		return loaded.Reason, nil
	case nil:
		return "", nil
	}
	panic("unreachable")
}
