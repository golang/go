# Gopls release policy

Gopls releases follow [semver](http://semver.org), with major changes and new
features introduced only in new minor versions (i.e. versions of the form
`v*.N.0` for some N). Subsequent patch releases contain only cherry-picked
fixes or superficial updates.

In order to align with the
[Go release timeline](https://github.com/golang/go/wiki/Go-Release-Cycle#timeline),
we aim to release a new minor version of Gopls approximately every three
months, with patch releases approximately every month, according to the
following table:

| Month   | Version(s)   |
| ----    | -------      |
| Jan     | `v*.<N+0>.0` |
| Jan-Mar | `v*.<N+0>.*` |
| Apr     | `v*.<N+1>.0` |
| Apr-Jun | `v*.<N+1>.*` |
| Jul     | `v*.<N+2>.0` |
| Jul-Sep | `v*.<N+2>.*` |
| Oct     | `v*.<N+3>.0` |
| Oct-Dec | `v*.<N+3>.*` |

For more background on this policy, see https://go.dev/issue/55267.
