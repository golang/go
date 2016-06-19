tables.go: ../x86map/map.go ../x86.csv 
	go run ../x86map/map.go -fmt=decoder ../x86.csv >_tables.go && gofmt _tables.go >tables.go && rm _tables.go

