package rethinkgo

type Session struct {
}

func (s *Session) Run(query Exp) *int { return nil }

type List []interface{}

type Exp struct {
	args []interface{}
}

func (e Exp) UseOutdated(useOutdated bool) Exp {
	return Exp{args: List{e, useOutdated}}
}
