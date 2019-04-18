package apidiff

import (
	"bytes"
	"fmt"
	"io"
)

// Report describes the changes detected by Changes.
type Report struct {
	Changes []Change
}

// A Change describes a single API change.
type Change struct {
	Message    string
	Compatible bool
}

func (r Report) messages(compatible bool) []string {
	var msgs []string
	for _, c := range r.Changes {
		if c.Compatible == compatible {
			msgs = append(msgs, c.Message)
		}
	}
	return msgs
}

func (r Report) String() string {
	var buf bytes.Buffer
	if err := r.Text(&buf); err != nil {
		return fmt.Sprintf("!!%v", err)
	}
	return buf.String()
}

func (r Report) Text(w io.Writer) error {
	if err := r.TextIncompatible(w, true); err != nil {
		return err
	}
	return r.TextCompatible(w)
}

func (r Report) TextIncompatible(w io.Writer, withHeader bool) error {
	if withHeader {
		return r.writeMessages(w, "Incompatible changes:", r.messages(false))
	}
	return r.writeMessages(w, "", r.messages(false))
}

func (r Report) TextCompatible(w io.Writer) error {
	return r.writeMessages(w, "Compatible changes:", r.messages(true))
}

func (r Report) writeMessages(w io.Writer, header string, msgs []string) error {
	if len(msgs) == 0 {
		return nil
	}
	if header != "" {
		if _, err := fmt.Fprintf(w, "%s\n", header); err != nil {
			return err
		}
	}
	for _, m := range msgs {
		if _, err := fmt.Fprintf(w, "- %s\n", m); err != nil {
			return err
		}
	}
	return nil
}
