from augment_self import augment_self

def call_augment(times_augmented, folder_link):
    for i in range(times_augmented):
        augment_self(folder_link)
        
    return 0


folder_link = 'data/yellow'
call_augment(4, folder_link)
    