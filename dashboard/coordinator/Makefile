coordinator: main.go
	GOOS=linux go build --tags=build_coordinator -o coordinator .

# After "make upload", either reboot the machine, or ssh to it and:
#   sudo systemctl restart gobuild.service
# And watch its logs with:
#   sudo journalctl -f -u gobuild.service
upload: coordinator
	cat coordinator | (cd buildongce && go run create.go --write_object=go-builder-data/coordinator)
