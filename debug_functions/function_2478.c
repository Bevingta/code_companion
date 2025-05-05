static inline void init_fs_pin(struct fs_pin *p, void (*kill)(struct fs_pin *))
{
	init_waitqueue_head(&p->wait);
	p->kill = kill;
}