<!-- These items need to be completed and moved to an appropriate location in the release notes. -->

<!-- go.dev/issue/61476, CL 541135 -->
TODO: The new `GORISCV64` environment variable needs to be documented. This note should be moved to an appropriate location in the release notes.

<!-- These items need to be reviewed, and mentioned in the Go 1.23 release notes if applicable.

None right now; more may come up later on.
-->

<!-- Maybe should be documented? Maybe shouldn't? Someone familiar with the change needs to determine.

CL 359594 ("x/website/_content/ref/mod: document dotless module paths") - resolved go.dev/issue/32819 ("cmd/go: document that module names without dots are reserved") and also mentioned accepted proposal go.dev/issue/37641
CL 570681 ("os: make FindProcess use pidfd on Linux") mentions accepted proposal go.dev/issue/51246 (described as fully implemented in Go 1.22) and NeedsInvestigation continuation issue go.dev/issue/62654.
CL 555075 ("x/tools/go/ssa: support range-over-func") - x/tools CL implements range-over-func support in x/tools/go/ssa for accepted proposal https://go.dev/issue/66601; this particular proposal and change doesn't seem to need a dedicated mention in Go 1.23 release notes but someone more familiar should take another look
-->

<!-- Items that don't need to be mentioned in Go 1.23 release notes but are picked up by relnote todo.

CL 458895 - an x/playground fix that mentioned an accepted cmd/go proposal go.dev/issue/40728 in Go 1.16 milestone...
CL 582097 - an x/build CL working on relnote itself; it doesn't need a release note
CL 561935 - crypto CL that used purego tag and mentioned accepted-but-not-implemented proposal https://go.dev/issue/23172 to document purego tag; doesn't need a release note
CL 568340 - fixed a spurious race in time.Ticker.Reset (added via accepted proposal https://go.dev/issue/33184), doesn't seem to need a release note
CL 562619 - x/website CL documented minimum bootstrap version on go.dev, mentioning accepted proposals go.dev/issue/54265 and go.dev/issue/44505; doesn't need a release note
CL 557055 - x/tools CL implemented accepted proposal https://go.dev/issue/46941 for x/tools/go/ssa
CL 564275 - an x/tools CL that updates test data in preparation for accepted proposal https://go.dev/issue/51473; said proposal isn't implemented for Go 1.23 and so it doesn't need a release note
CL 572535 - used "unix" build tag in more places, mentioned accepted proposal https://go.dev/issue/51572; doesn't need a release note
CL 555255 - an x/tools CL implements accepted proposal https://go.dev/issue/53367 for x/tools/go/cfg
CL 585216 - an x/build CL mentions accepted proposal https://go.dev/issue/56001 because it fixed a bug causing downloads not to be produced for that new-to-Go-1.22 port; this isn't relevant to Go 1.23 release notes
CL 481062 - added examples for accepted proposal https://go.dev/issue/56102; doesn't need a release note
CL 497195 - an x/net CL adds one of 4 fields for accepted proposal https://go.dev/issue/57893 in x/net/http2; seemingly not related to net/http and so doesn't need a Go 1.23 release note
CL 463097, CL 568198 - x/net CLs that implemented accepted proposal https://go.dev/issue/57953 for x/net/websocket; no need for rel note
many x/net CLs - work on accepted proposal https://go.dev/issue/58547 to add a QUIC implementation to x/net/quic
CL 514775 - implements a performance optimization for accepted proposal https://go.dev/issue/59488
CL 484995 - x/sys CL implements accepted proposal https://go.dev/issue/59537 to add x/sys/unix API
CL 555597 - optimizes TypeFor (added in accepted proposal https://go.dev/issue/60088) for non-interface types; doesn't seem to need a release note
a few x/tools CLs deprecated and deleted the experimental golang.org/x/tools/cmd/getgo tool per accepted proposal https://go.dev/issue/60951; an unreleased change and not something that's in scope of Go 1.23 release notes
many x/vuln CLs to implement accepted proposal https://go.dev/issue/61347 ("x/vuln: convert govulncheck output to sarif format") in govulncheck
CL 516355 - x/crypto CL that implemented accepted proposal https://go.dev/issue/61447 for x/crypto/ssh; doesn't need a Go 1.23 release note
CL 559799 - a Go 1.22 release note edit CL mentioned a Go 1.22 accepted proposal https://go.dev/issue/62039, a little after Go 1.23 development began
CL 581555 - an x/tools CL mentioned accepted proposal https://go.dev/issue/62292 for x/tools/go/aalysis; doesn't need a Go 1.23 release note
CL 578355 - mentioned accepted proposal https://go.dev/issue/63131 to add GOARCH=wasm32, but that proposal hasn't been implemented in Go 1.23 so it doesn't need a release note
CL 543335 - x/exp CL that backported a change to behavior in slices package (accepted proposal https://go.dev/issue/63393) to x/exp/slices; doesn't need a Go 1.23 release note
CL 556820 - x/tools CL implemented accepted proposal https://go.dev/issue/64548 for x/tools/go/analysis
CL 557056 - x/tools CL implemented accepted proposal https://go.dev/issue/64608 for x/tools/go/packages
CL 558695 - x/crypto CL worked on accepted proposal https://go.dev/issue/64962 for x/crypto/ssh
CL 572016 - x/tools CL implemented accepted proposal https://go.dev/issue/65754 for x/tools/go/cfg
a few x/tools CLs tagged and deleted the golang.org/x/tools/cmd/guru command per accepted proposal https://go.dev/issue/65880; an unreleased change and not something that's in scope of Go 1.23 release notes
CL 580076 - seemingly internal cmd/go change to propagate module information for accepted proposal https://go.dev/issue/66315; doesn't seem to warrant a release note
CL 529816 - the 'tests' vet check was initially added to the 'go test' suite per accepted proposal https://go.dev/issue/44251, but the change was rolled back in CL 571695, with no roll forward as of 2024-05-23; nothing to document in Go 1.23 release notes for it at this time
CL 564035 - changed encoding/xml, but the change was too disrptive and rolled back in CL 570175, reopening tracking issue go.dev/issue/65691; nothing to document in Go 1.23 release notes
CL 587855 - a demonstration of benefit of accepted proposal https://go.dev/issue/60529; actual change isn't happening in Go 1.23 so doesn't need a release note
CL 526875 - x/crypto CL implemented accepted proposal https://go.dev/issue/62518 for x/crypto/ssh
-->
