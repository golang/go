tables.go: ../s390xmap/map.go ../s390x.csv
	go run ../s390xmap/map.go -fmt=decoder ../s390x.csv >_tables.go && gofmt _tables.go >tables.go && rm _tables.go
