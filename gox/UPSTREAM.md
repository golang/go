# Upstream source provenance

The internal parser and scanner are adapted from Go sources at repository commit
64f1787400cb9507e355a13bf5f25c7158dabe4d. Original BSD notices and LICENSE remain.

Changes: internal scanner import; GoX opt-in scanner/parser extensions; lowering
support. The newer optional ast.BasicLit.ValueEnd fields are omitted so the
standalone module can build with Go 1.26. Scanner.End remains available internally.
Ordinary Go parsing is retained when GoX mode is off. Upstream test adaptations
and compatibility tests must accompany changes to the copied implementation.
