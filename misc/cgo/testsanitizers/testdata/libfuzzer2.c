#include <stddef.h>

#include "libfuzzer2.h"

int LLVMFuzzerTestOneInput(char *data, size_t size) {
 	if (size > 0 && data[0] == 'H')
		if (size > 1 && data[1] == 'I')
			if (size > 2 && data[2] == '!')
				FuzzMe(data, size);
	return 0;
}
