// clang -gdwarf-5 -O2 -nostdlib

__attribute__((noinline, cold))
static int sum(int i) {
  int j, s;

  s = 0;
  for (j = 0; j < i; j++) {
    s += j * i;
  }
  return s;
}

int main(int argc, char** argv) {
  if (argc == 0) {
    return 0;
  }
  return sum(argc);
}
