
class DataWrapper:

    def __init__(self):
        self.training = Data()
        self.test = Data()


class Data:

    def __init__(self):
        self.images = ['um', 'dois', 'três', 'quatro', 'cinco']
        self.labels = [1, 2, 3, 4, 5]
        self.current_index = 0

    def is_batch_size_valid(self, batch_size):
        return self.current_index + batch_size > len(self.images)

    def create_joined_list(self, batch_size, source_list, update_index = False):
        first_part = source_list[self.current_index : len(source_list)]
        last_part = source_list[0 : batch_size - len(first_part)]
        if update_index:
            self.current_index = batch_size - len(first_part)
        return first_part + last_part

    def next_batch(self, batch_size):

        images_batch = None
        labels_batch = None

        #TODO verificar batch maior que o tamanho da lista.
        if self.is_batch_size_valid(batch_size):

            images_batch = self.create_joined_list(batch_size, self.images)
            labels_batch = self.create_joined_list(batch_size, self.labels, True)

        else:

            final_index = self.current_index + batch_size
            images_batch = self.images[self.current_index : final_index]
            labels_batch = self.labels[self.current_index : final_index]

            self.current_index = final_index

        return images_batch, labels_batch

    def load_labels(self, path):
        print("not implemented yet")

    def load_images(self, path):
        print("not implemented yet")
