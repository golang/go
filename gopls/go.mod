module golang.org/x/tools/gopls

go 1.18

require (
	github.com/google/go-cmp v0.5.9
	github.com/jba/printsrc v0.2.2
	github.com/jba/templatecheck v0.6.0
	github.com/sergi/go-diff v1.1.0
	golang.org/x/mod v0.7.0
	golang.org/x/sync v0.1.0
	golang.org/x/sys v0.2.0
	golang.org/x/text v0.4.0
	golang.org/x/tools v0.2.1-0.20221108172846-9474ca31d0df
	golang.org/x/vuln v0.0.0-20221109205719-3af8368ee4fe
	gopkg.in/yaml.v3 v3.0.1
	honnef.co/go/tools v0.3.3
	mvdan.cc/gofumpt v0.4.0
	mvdan.cc/xurls/v2 v2.4.0
)

require golang.org/x/exp v0.0.0-20221031165847-c99f073a8326 // indirect

require (
	github.com/BurntSushi/toml v1.2.1 // indirect
	github.com/google/safehtml v0.1.0 // indirect
	golang.org/x/exp/typeparams v0.0.0-20221031165847-c99f073a8326 // indirect
)

replace golang.org/x/tools => ../
