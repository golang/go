package enum

type Result enum {
	Allow
	Deny { Reason string }
}
