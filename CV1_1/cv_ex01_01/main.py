# Solution by 
# Emilio Brambilla
# Moritz Lahann
# Lasse Haffke

import numpy as np
import matplotlib.pyplot as plt
import os

test_batch = "/home/lasse/src/cv_exercises/cv_ex01_01/cifar-10-batches-py/test_batch"
train_batch = "/home/lasse/src/cv_exercises/cv_ex01_01/cifar-10-batches-py/data_batch_1"
project_folder = "/home/lasse/src/cv_exercises/cv_ex01_01/"
histogram_output_folder = "histograms/bin_5"


def main():
    test_file_folders = ["test_automobile", "test_deer", "test_ship"]
    train_file_folders = ["automobile", "deer", "ship"]
    label_numbers = [1, 4, 8]

    # create_and_save_histogram(test_batch, label_numbers, test_file_folders, 10)
    # create_and_save_histogram(train_batch, label_numbers, train_file_folders, 30)
    trained_histograms = read_in_histograms(project_folder, train_file_folders)
    test_histograms = read_in_histograms(project_folder, test_file_folders)

    classification_results = classify_against(test_histograms, trained_histograms)
    classifier_accuracy = calculate_accuracy(classification_results)
    print("classification accuracy = {}".format(classifier_accuracy))


def create_and_save_histogram(batch_path, label_numbers, output_directories, max_images):
    reader = BatchFileReader()
    test_data = reader.read(batch_path)
    for number, name in zip(label_numbers, output_directories):
        data = get_index_of(test_data[b"labels"], number, max_images)
        save_histogram(data, test_data[b"data"], f"{name}")


def save_histogram(indexes, data, class_name):
    writer = BatchFileWriter()
    for index in indexes:
        img_gray = convert_to_grayscale(get_colors_of_image_at(data, index))
        weights, bins = calculate_histogram(img_gray)
        writer.write(weights, os.path.join(project_folder, histogram_output_folder, f"{class_name}/{index}"))


def calculate_histogram(grayscaled_image):
    bin_size = 5
    bins = np.arange(0, 256, bin_size)
    weights, bins, _ = plt.hist(grayscaled_image, bins=bins)
    return weights, bins


def classify_against(test_histograms,trained_histograms):
    classification = {}
    for test in test_histograms.items():
        nearest_neighbour = None
        min_distance = float("inf")
        for trained in trained_histograms.items():
            # numpy function for the euclidean distance
            dist = np.linalg.norm(test[1] - trained[1])
            if dist < min_distance:
                nearest_neighbour = trained
                min_distance = dist
        classification[test[0]] = nearest_neighbour[0]
    return classification


def calculate_accuracy(results):
    correct_classified_counter = 0
    for tested_histogram, nearest_neighbour in results.items():
        if get_label(tested_histogram) == get_label(nearest_neighbour):
            correct_classified_counter += 1
    classifier_accuracy = correct_classified_counter / len(results)
    return classifier_accuracy


def read_in_histograms(project_folder, sub_directories):
    histograms = {}
    reader = BatchFileReader()
    for train_file in sub_directories:
        train_file_folder_path = os.path.join(project_folder, histogram_output_folder, train_file)
        for file in os.listdir(train_file_folder_path):
            file_full_path = os.path.join(train_file_folder_path, file)
            histograms[file_full_path] = reader.read(file_full_path)
    return histograms


def get_label(path):
    if "automobile" in path:
        return 1
    elif "deer" in path:
        return 4
    elif "ship" in path:
        return 8
    else:
        return -1







def convert_to_grayscale(image_dictionary):
    red = image_dictionary["red"]
    green = image_dictionary["green"]
    blue = image_dictionary["blue"]
    grayscaled = []
    for pixel in range(len(red)):
        gray_value = (float(green[pixel]) + float(red[pixel]) + float(blue[pixel])) / 3
        grayscaled += [gray_value]
    return grayscaled



def get_colors_of_image_at(data, index):
    raw_data = data[index, :]
    red = raw_data[:1024]
    green = raw_data[1024:2048]
    blue = raw_data[2048:]
    return {"red": red, "green": green, "blue": blue}

def get_index_of(data, label, max):
    res = []
    for index, item in enumerate(data):
        if item == label:
            res += [index]
            if len(res) == max:
                break
    return res

# IO classes
class BatchFileReader(object):

    def read(self, file_path):
        import pickle
        with open(file_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict

class BatchFileWriter(object):
    def write(self, obj, path):
        import pickle
        import os
        dirname = os.path.dirname(path)
        os.makedirs(dirname,exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(obj, file)

if __name__ == "__main__":
    main()
