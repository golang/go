newdecode.txt:
	cd ..; go test -cover -run 'ObjdumpARMCond' -v -timeout 10h -printtests -long 2>&1 | tee log
	cd ..; go test -cover -run 'ObjdumpARMUncond' -v -timeout 10h -printtests -long 2>&1 | tee -a log
	egrep '	(gnu|plan9)	' ../log |sort >newdecode.txt

