package model

type Decision enum {
	Allow
	Deny { Reason string }
}
