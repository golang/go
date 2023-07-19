module golang.org/x/tools/gopls

go 1.18

require (
	github.com/google/go-cmp v0.5.9
	github.com/jba/printsrc v0.2.2
	github.com/jba/templatecheck v0.6.0
	github.com/sergi/go-diff v1.1.0
	golang.org/x/mod v0.12.0
	golang.org/x/sync v0.3.0
	golang.org/x/sys v0.10.0
	golang.org/x/telemetry v0.0.0-20230728182230-e84a26264b60
	golang.org/x/text v0.11.0
	golang.org/x/tools v0.6.0
	golang.org/x/vuln v0.0.0-20230110180137-6ad3e3d07815
	gopkg.in/yaml.v3 v3.0.1
	honnef.co/go/tools v0.4.2
	mvdan.cc/gofumpt v0.4.0
	mvdan.cc/xurls/v2 v2.4.0
)

require (
	github.com/BurntSushi/toml v1.2.1 // indirect
	github.com/google/safehtml v0.1.0 // indirect
	golang.org/x/exp v0.0.0-20220722155223-a9213eeb770e // indirect
	golang.org/x/exp/typeparams v0.0.0-20221212164502-fae10dda9338 // indirect
)

replace golang.org/x/tools => ../
