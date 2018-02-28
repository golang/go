go test command:
	cd ..; go test -run 'ObjdumpARM64Cond' -v -timeout 10h -long 2>&1 | tee log
	cd ..; go test -run 'ObjdumpARM64Testdata' -v -timeout 10h -long 2>&1 | tee -a log
	cd ..; go test -run 'ObjdumpARM64' -v -timeout 10h -long 2>&1 | tee -a log
	cd ..; go test -run 'ObjdumpARM64Manual' -v -timeout 10h -long 2>&1 | tee -a log
	cd ..; go test -run 'TestDecode'
	cd ..; go test -run '.*'
