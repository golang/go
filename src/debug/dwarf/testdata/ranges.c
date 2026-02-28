// gcc -g -O2 -freorder-blocks-and-partition

const char *arr[10000];
const char *hot = "hot";
const char *cold = "cold";

__attribute__((noinline))
void fn(int path) {
	int i;

	if (path) {
		for (i = 0; i < sizeof arr / sizeof arr[0]; i++) {
			arr[i] = hot;
		}
	} else {
		for (i = 0; i < sizeof arr / sizeof arr[0]; i++) {
			arr[i] = cold;
		}
	}
}

int main(int argc, char *argv[]) {
	fn(argc);
	return 0;
}
