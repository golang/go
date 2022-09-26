module golang.org/x/tools/gopls

go 1.18

require (
	github.com/google/go-cmp v0.5.8
	github.com/jba/printsrc v0.2.2
	github.com/jba/templatecheck v0.6.0
	github.com/sergi/go-diff v1.1.0
	golang.org/x/mod v0.6.0-dev.0.20220419223038-86c51ed26bb4
	golang.org/x/sync v0.0.0-20220722155255-886fb9371eb4
	golang.org/x/sys v0.0.0-20220808155132-1c4a2a72c664
	golang.org/x/text v0.3.7
	golang.org/x/tools v0.1.13-0.20220928184430-f80e98464e27
	golang.org/x/vuln v0.0.0-20221010193109-563322be2ea9
	gopkg.in/yaml.v3 v3.0.1
	honnef.co/go/tools v0.3.3
	mvdan.cc/gofumpt v0.3.1
	mvdan.cc/xurls/v2 v2.4.0
)

require golang.org/x/exp v0.0.0-20220722155223-a9213eeb770e // indirect

require (
	github.com/BurntSushi/toml v1.2.0 // indirect
	github.com/google/safehtml v0.0.2 // indirect
	golang.org/x/exp/typeparams v0.0.0-20220722155223-a9213eeb770e // indirect
)

replace golang.org/x/tools => ../
