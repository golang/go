typedef struct Issue4339 Issue4339;

struct Issue4339 {
	char *name;
	void (*bar)(void);
};

extern Issue4339 exported4339;
void	handle4339(Issue4339*);
