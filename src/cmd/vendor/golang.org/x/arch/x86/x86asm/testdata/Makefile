libmach8db: libmach8db.c
	9c libmach8db.c && 9l -o libmach8db libmach8db.o; rm libmach8db.o

newdecode.txt:
	cd ..; go test -cover -run 'Objdump.*32' -v -timeout 10h -printtests 2>&1 | tee log
	cd ..; go test -cover -run 'Objdump.*64' -v -timeout 10h -printtests 2>&1 | tee -a log
	cd ..; go test -cover -run 'Xed.*32' -v -timeout 10h -printtests 2>&1 | tee -a log
	cd ..; go test -cover -run 'Xed.*64' -v -timeout 10h -printtests 2>&1 | tee -a log
	cd ..; go test -cover -run 'Plan9.*32' -v -timeout 10h -printtests 2>&1 | tee -a log
	cd ..; go test -cover -run 'Plan9.*64' -v -timeout 10h -printtests 2>&1 | tee -a log
	egrep '	(gnu|intel|plan9)	' ../log |sort >newdecode.txt

