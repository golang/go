tables.go: ../armmap/map.go ../arm.csv 
	go run ../armmap/map.go -fmt=decoder ../arm.csv >_tables.go && gofmt _tables.go >tables.go && rm _tables.go
