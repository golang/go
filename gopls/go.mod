module golang.org/x/tools/gopls

go 1.18

require (
	github.com/google/go-cmp v0.5.8
	github.com/jba/printsrc v0.2.2
	github.com/jba/templatecheck v0.6.0
	github.com/sergi/go-diff v1.1.0
	golang.org/x/mod v0.6.0-dev.0.20220419223038-86c51ed26bb4
	golang.org/x/sys v0.0.0-20220722155257-8c9f86f7a55f
	golang.org/x/tools v0.1.11
	golang.org/x/vuln v0.0.0-20220725105440-4151a5aca1df
	honnef.co/go/tools v0.3.2
	mvdan.cc/gofumpt v0.3.1
	mvdan.cc/xurls/v2 v2.4.0
)

require (
	github.com/BurntSushi/toml v1.2.0 // indirect
	github.com/google/safehtml v0.0.2 // indirect
	golang.org/x/exp/typeparams v0.0.0-20220722155223-a9213eeb770e // indirect
	golang.org/x/sync v0.0.0-20220722155255-886fb9371eb4
	golang.org/x/text v0.3.7 // indirect
)

replace golang.org/x/tools => ../
