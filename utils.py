import cv2
import os
import moviepy.editor as mp
import matplotlib.pyplot as plt
import glob
import dlib
import numpy as np
import enlighten
import torch
import time
import random

ROOT_PATH = "temp"

# Extract images from given video file name
def extractImages(file_name):
    cap = cv2.VideoCapture(file_name)
    i=0
    image_array = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_array.append(frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    return image_array

# Get audio from video
def extractAudio(file_name, save_path):
    if save_path[-1] == '/':
        save_path = save_path[:-1]
    clip = mp.VideoFileClip(file_name)
    save_name = file_name.split('/')[-1]
    clip.audio.write_audiofile(f"{save_path + '/' + save_name[:-4]}.wav")

def create_audio_dataset(videos_dir_path, save_path):
    if videos_dir_path[-1] != '/':
        videos_dir_path += '/'
    if save_path[-1] != '/':
        save_path += '/'
    TRUTH_PATH = videos_dir_path + 'Truths'
    LIES_PATH = videos_dir_path + 'Lies'
    
    for vid in glob.glob( TRUTH_PATH + "/*.mp4"):
        extractAudio(vid, save_path + 'Truth')
    for vid in glob.glob( LIES_PATH + "/*.mp4"):
        extractAudio(vid, save_path + 'Lies')

# Display a list of images
def plotImages(images_arr,image_n):
    fig,axes = plt.subplots(1,image_n,figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Load images from a folder to memory
def getImageArray(folder_name):
    cv_img = []
    for img in glob.glob(folder_name + "/*.jpg"):
        n = cv2.imread(img)
        n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
        cv_img.append(n)
    return cv_img

# Get face crop dimensions
face_detector = dlib.get_frontal_face_detector()
def get_crop_dim(image, MARGIN):
    try:
        box=face_detector.run(image)[0][0]
    except:
        raise Exception("face error")
    left, top, right, bottom = box.left() - 90 - MARGIN, box.top() - 150 - MARGIN, box.right() + 90 + MARGIN, box.bottom() + 30 + MARGIN
    if left < 0: left = 0
    elif left > 1280: left = 1280
    if right < 0: right = 0
    elif right > 1280: right = 1280
    if top < 0: top = 0
    elif top > 720: top=720
    if bottom < 0: bottom = 0
    elif bottom > 720: bottom=720
    
    return left, top, right, bottom

# Create sequence of face images from an image list 
def get_sequences(image_list):
    '''Image list: A list of images loaded using openCV and in RGB'''
    seq = []
    img_id = 0
    while True:
        try:
            left, top, right, bottom = get_crop_dim(image_list[img_id], 50)
            break
        except:
            img_id += 1
            if img_id >= len(image_list):
                print("Video skipped, face not detected.")
                return None
            continue

    for image in image_list:
        #print(top,bottom,left,right)
        cropped_img = image[top:bottom,left:right]
        img_small = cv2.resize(cropped_img, (224,224))
        seq.append(img_small)
    return seq

from tqdm import tqdm
import sys
# Function to load sequences of training data
def load_train_data(path):
    if path[-1] != '/':
        path += '/'

    # Clear dataset directory
    import os
    os.system('rm -r dataset')
    os.system('mkdir dataset')
    os.system('mkdir dataset/Truth')
    os.system('mkdir dataset/Lie')

    truth_path = path + 'Truth'
    lies_path = path + 'Lie'
    DATASET_WRITE_DIR = 'dataset'
    labels = []
    sequences = []
    truth_glob = glob.glob(truth_path    + "/*.mp4")
    lies_glob = glob.glob(lies_path    + "/*.mp4")
    with tqdm(total=len(truth_glob)) as progress_bar:
        for idx, video in enumerate(truth_glob):
            # Preprocess frames
            image_list = extractImages(video)
            cropped_images = get_sequences(image_list)
            if cropped_images:
                sequences.append(video)
                labels.append('Truth')
            os.system('mkdir ' + DATASET_WRITE_DIR + '/Truth/' + str(idx))
            # Write back frames
            for idx_img, image in enumerate(cropped_images):
                cv2.imwrite(DATASET_WRITE_DIR + '/Truth/' + str(idx) + '/' + str(idx_img) + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Write back MFCC
            devnull = open(os.devnull, "w")
            old_stdout = sys.stdout
            sys.stdout = devnull
            clip = mp.VideoFileClip(video)
            clip.audio.write_audiofile(DATASET_WRITE_DIR + '/Truth/' + str(idx) + '/audio.wav')
            sys.stdout = old_stdout
            devnull.close()
            progress_bar.update(1)
            # break

    with tqdm(total=len(lies_glob)) as progress_bar:
        for idx, video in enumerate(lies_glob):
            image_list = extractImages(video)
            cropped_images = get_sequences(image_list)
            if cropped_images:
                sequences.append(video)
                labels.append('Lie')
            os.system('mkdir ' + DATASET_WRITE_DIR + '/Lie/' + str(idx))
            for idx_img, image in enumerate(cropped_images):
                cv2.imwrite(DATASET_WRITE_DIR + '/Lie/' + str(idx) + '/' + str(idx_img) + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Write back MFCC
            devnull = open(os.devnull, "w")
            old_stdout = sys.stdout
            sys.stdout = devnull
            clip = mp.VideoFileClip(video)
            clip.audio.write_audiofile(DATASET_WRITE_DIR + '/Lie/' + str(idx) + '/audio.wav')
            sys.stdout = old_stdout
            devnull.close()
            progress_bar.update(1)
            # break

    return sequences, labels

from torchvision import transforms
def load_dataset(dataset_dir, as_tensors=True):
    if dataset_dir[-1] != '/':
        dataset_dir += '/'
    TRUTH_ROOT = dataset_dir + 'Truth'
    LIES_ROOT = dataset_dir + 'Lie'
    # All items in truth
    TRUTH_GLOB = glob.glob(TRUTH_ROOT + '/*')
    LIES_GLOB = glob.glob(LIES_ROOT + '/*')
    sequences = []
    labels = []
    to_tensor = transforms.ToTensor()
    with tqdm(total=len(TRUTH_GLOB)+len(LIES_GLOB), desc="Loading data...") as progress_bar:
        for idx, seq_dir in enumerate(TRUTH_GLOB):
            seq_imgs = []
            for img in glob.glob(seq_dir + '/*.jpg'):
                # Load each image
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if as_tensors:
                    img = to_tensor(img)
                seq_imgs.append(img)
            sequences.append(seq_imgs)
            labels.append('Truth')
            progress_bar.update(1)
        for idx, seq_dir in enumerate(LIES_GLOB):
            seq_imgs = []
            for img in glob.glob(seq_dir + '/*.jpg'):
                # Load each image
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if as_tensors:
                    img = to_tensor(img)
                seq_imgs.append(img)
            sequences.append(seq_imgs)
            labels.append('Lie')
            progress_bar.update(1)

    return sequences, labels

# Detect GPU
if torch.cuda.is_available():
    print("\033[93mGPU found, using CUDA !\033[0m")
    device = 'cuda'
else:
    print("\033[91mGPU NOT found, using CPU !\033[0m")
    device = 'cpu'

LABELS = {"Truth": 0, "Lie": 1}
INV_LABELS = {0:"Truth", 1:"Lie"}

def train_test_split(dataset_dir, train_percent=80, validate_percent=0, class_equalize = True):
    print("\033[91mWarning: Dataset randomized!\033[0m")
    if train_percent < 0 or validate_percent < 0 or train_percent + validate_percent > 100:
        raise Exception("Invalid split percent!")
    
    # if train_percent + validate_percent == 100
    # Enumerate directories
    if dataset_dir[-1] != '/':
        dataset_dir += '/'
    TRUTH_ROOT = dataset_dir + 'Truth'
    LIES_ROOT = dataset_dir + 'Lie'
    TRUTH_GLOB = glob.glob(TRUTH_ROOT + '/*')
    LIE_GLOB = glob.glob(LIES_ROOT + '/*')
    # Shuffle and split from truth and lie
    random.shuffle(TRUTH_GLOB)
    random.shuffle(LIE_GLOB)
    if class_equalize:
        class_len = min(len(TRUTH_GLOB), len(LIE_GLOB))
        TRUTH_GLOB = TRUTH_GLOB[:class_len]
        LIE_GLOB = LIE_GLOB[:class_len]

    TRUTH_TRAIN_PCT = int((train_percent/100) * len(TRUTH_GLOB))
    LIE_TRAIN_PCT = int((train_percent/100) * len(LIE_GLOB))

    TRUTH_VALID_PCT = int((validate_percent/100) * len(TRUTH_GLOB)) + TRUTH_TRAIN_PCT
    LIE_VALID_PCT = int((validate_percent/100) * len(LIE_GLOB)) + LIE_TRAIN_PCT

    if TRUTH_TRAIN_PCT == 0 or LIE_TRAIN_PCT == 0:
        print("\033[93mWarn: 0 items in training set (for at least one class)\033[0m")

    if TRUTH_VALID_PCT == TRUTH_TRAIN_PCT or LIE_VALID_PCT == LIE_TRAIN_PCT:
        print("\033[93mWarn: 0 items in validation set (for at least one class)\033[0m")

    if TRUTH_VALID_PCT == len(TRUTH_GLOB) or LIE_VALID_PCT == len(LIE_GLOB):
        print("\033[93mWarn: 0 items in test set (for at least one class)\033[0m")

    truth_train = TRUTH_GLOB[:TRUTH_TRAIN_PCT]
    lie_train = LIE_GLOB[:LIE_TRAIN_PCT]

    truth_validate = TRUTH_GLOB[TRUTH_TRAIN_PCT:TRUTH_VALID_PCT]
    lie_validate = LIE_GLOB[LIE_TRAIN_PCT:LIE_VALID_PCT]

    truth_test = TRUTH_GLOB[TRUTH_VALID_PCT:]
    lie_test = LIE_GLOB[LIE_VALID_PCT:]

    train_set = truth_train + lie_train
    train_labels = [LABELS["Truth"]] * len(truth_train) + [LABELS["Lie"]] * len(lie_train)
    valid_set = truth_validate + lie_validate
    valid_labels = [LABELS["Truth"]] * len(truth_validate) + [LABELS["Lie"]] * len(lie_validate)
    test_set = truth_test + lie_test
    test_labels = [LABELS["Truth"]] * len(truth_test) + [LABELS["Lie"]] * len(lie_test)

    # Shuffle one more time
    try:
        temp = list(zip(train_set, train_labels))
        random.shuffle(temp)
        train_set, train_labels = zip(*temp)
    except:
        train_set = []
        train_labels = []
    try:
        temp = list(zip(valid_set, valid_labels))
        random.shuffle(temp)
        valid_set, valid_labels = zip(*temp)
    except:
        valid_set = []
        valid_labels = []
    try:
        temp = list(zip(test_set, test_labels))
        random.shuffle(temp)
        test_set, test_labels = zip(*temp)
    except:
        test_set = []
        test_labels = []

    print("Split\n        Train\tValid\tTest")
    print("Truth:  ", len(truth_train), "\t ", len(truth_validate), "\t ", len(truth_test))
    print("Lies:   ", len(lie_train), "\t ", len(lie_validate), "\t ", len(lie_test))
    print("Total:  ", len(train_set), "\t ", len(valid_set), "\t ", len(test_set))

    return train_set, train_labels, valid_set, valid_labels, test_set, test_labels

def get_dataset(directory, dataset , class_equalize=False, googlenet_model = False, train_percent = 80, validate_percent = 0):
    train_files, train_labels, valid_files, valid_labels, test_files, test_labels = train_test_split(directory, class_equalize=class_equalize, train_percent=train_percent, validate_percent=validate_percent)
    valid_set = dataset(valid_files, valid_labels, googlenet_model=googlenet_model)
    train_set = dataset(train_files, train_labels, googlenet_model=googlenet_model)
    test_set = dataset(test_files, test_labels, googlenet_model=googlenet_model)
    return train_set, valid_set, test_set

def get_dataloader(directory, dataset, class_equalize=False, shuffle=True, batch_size=1, googlenet_model = False, train_percent=80, validate_percent=0):
    train_set, valid_set, test_set = get_dataset(directory, dataset, class_equalize, googlenet_model, train_percent=train_percent, validate_percent=validate_percent)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    if(len(valid_set) != 0):
        valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    else:
        valid_dataloader = None
    if(len(test_set) != 0):
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    else:
        test_dataloader = None
    return train_dataloader, valid_dataloader, test_dataloader

# Function to check accuracy
def validate(model, dataloader, PROB_THRESHOLD=0, verbose = False):
    # Switch model to evaluation mode
    model.eval()
    # Switch dataloader/dataset to test mode
    try:
        validation_progress.count = 0
        validation_progress.start = time.time()
        validation_progress.total = len(dataloader)
    except:
        pass
    tn = tp = fn = fp = 0
    incorrect = 0
    
    print("Prediction: ", end="")
    prob = []
    for batch in dataloader:
        # Get test sequence
        input_seq, target_seq = batch

        with torch.no_grad():
            output = model(input_seq)
        # Iterate through outputs for each batch
        # Positive=> Lie, Negative=> Truth
        for id, batch_op in enumerate(output):
            if (batch_op[0] > batch_op[1] + PROB_THRESHOLD) and target_seq[id] == 0:     # Class 0, truth => correct prediction
                print("\033[92m{}\033[0m".format(INV_LABELS[0][0]), end="")
                tn += 1
            elif (batch_op[0] < batch_op[1] - PROB_THRESHOLD) and target_seq[id] == 1:   # Class 1, lie => correct prediction
                print("\033[92m{}\033[0m".format(INV_LABELS[1][0]), end="")
                tp += 1
            elif (batch_op[0] > batch_op[1] + PROB_THRESHOLD):                           # 0 predicted, correct: 1 , Lie predicted as Truth
                print("\033[91m{}\033[0m".format(INV_LABELS[0][0]), end="")
                prob.append(round(batch_op[0].item() - batch_op[1].item(), 4))
                fn += 1
            elif (batch_op[0] < batch_op[1] - PROB_THRESHOLD):                           # 1 predicted, correct: 0 , Truth predicted as lie
                print("\033[91m{}\033[0m".format(INV_LABELS[1][0]), end="")
                prob.append(round(batch_op[0].item() - batch_op[1].item(), 4))
                fp += 1
            else:                                                       # Equal probabilities
                print("\033[93mU\033[0m", end="")
                incorrect += 1
        
        # Delete tensors to free memory
        #del seq_len
        del input_seq
        del target_seq
        del output
        torch.cuda.empty_cache()
        try:
            validation_progress.update()
        except:
            pass

    # !!! Do not forget to switch back to train mode before training !!!
    model.train()
    try:
        accuracy = float((tn+tp)/(fn+fp+tn+tp+incorrect))
        f1 = float(tp/(tp + 0.5*(fp+fn)))
    except:
        accuracy = 0
    print(", accuracy: {:.4f}".format(accuracy * 100), end="")
    if verbose:
        try:
            print(", F1: {:.4f}, TP:{},TN:{},FP:{},FN:{}".format(f1,tp,tn,fp,fn))
        except:
            pass
        print("Probabilities for misclassifications off by:", prob)
    else:
        print()
    return accuracy


# Plotter
from matplotlib import pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import Output
# %matplotlib inline

class plot_line:
    def __init__(self, num_lines = 1):
        self.x_index = []
        self.x_data = []
        self.y_data = []
        for i in range(num_lines):
            self.x_index.append(0)
            self.x_data.append([])
            self.y_data.append([])
        self.num_lines = num_lines
        global out
        plt.figure(figsize=(18, 4), dpi=80)
        out = Output()
        display(out)
    def plot(self):
        cols = ["r-", "g-", "b-"]
        with out:
            for i in range(self.num_lines):
                plt.plot(self.x_data[i], self.y_data[i], cols[i % len(cols)])
            clear_output(wait=True)
            display(plt.gcf())
        # plt.draw()
    def update(self, new_data, line_id = 0,update_time = True, display = False):
        self.x_index[line_id] += 1
        self.x_data[line_id].append(self.x_index[line_id])
        self.y_data[line_id].append(new_data)
        if display:
            self.plot()

def train(model, train_dataloader, test_dataloader=None, n_epochs=1, optimizer_obj=None, lr=1e-4, criterion=None, MAX_ACC=60, savefile_name="model", verbose=False, use_augmentation = False):
    '''
    model: The required model to train (should take only the input sequence as input)
    train_dataloader: PyTorch Dataloader for the training set
    test_dataloader: PyTorch Dataloader for the testing set
    n_epochs: Number of epochs to run
    optimizer: Optimizer to use (object)
    criterion: Loss criterion to use (object)
    MAX_ACC: Current max accuracy of the model on disk (Any instance of training model with accuracy greater than MAX_ACC will overwrite the model on disk)
    savefile_name: Filename for the max accuracy model to be saved on disk (Do not give file extension. Eg: savefile_name=model1)
    verbose: Whether to display additional statistiics, including tp,tn,fp,fn, f-score, and misclassification margins
    '''
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('Training...', color='black_on_peachpuff2', justify=enlighten.Justify.CENTER)
    status_bar2 = manager.status_bar('Waiting...', color='black_on_peachpuff2', justify=enlighten.Justify.CENTER)
    epoch_progress = manager.counter(total=n_epochs, desc="Epochs", unit="epochs", color="cyan")
    batches_progress = manager.counter(total=len(train_dataloader), desc="Batches", unit="batches", color="red")
    global validation_progress
    validation_progress = manager.counter(total=len(train_dataloader), desc="Validating...", unit="batches", color="green", leave=False)

    loss_plot = plot_line(2)
    avg_loss = 0
    acc = 0
    cur_max_acc = 0
    train_accuracy = []
    model.train()
    optimizer = optimizer_obj(model.parameters(),lr)
    for epoch in range(1, n_epochs + 1):
        # with tqdm(total=len(dataloader), desc="Training...") as progress_bar:
        batches_progress.count = 0
        batches_progress.start = time.time()
        # batches_progress.clear()

        epoch_loss = 0
        if use_augmentation:                            # Do not use this method!
            train_dataloader.dataset.augmentation()
        for i, video in enumerate(train_dataloader):
            optimizer.zero_grad() # Clear existing gradients

            # Get training batch from dataloader
            input_seq, target_seq = video

            status_bar.update("Training with:" + str([INV_LABELS[t.item()][0] for t in target_seq]))
            status_bar2.update("Accuracy: {:.4f}, Max: {:.4f}".format(acc, cur_max_acc) + ", Loss: {:.4f}".format(avg_loss/max(1,epoch-1)))

            # Get predictions
            output = model(input_seq)

            # Prepare target sequence on required device to calculate loss
            target_seq = target_seq.to(device, dtype=torch.long)
            loss = criterion(output, target_seq.view(-1).long())

            # Update weights
            loss.backward() # Backpropagation and gradient calculation
            optimizer.step() # Weight updation

            # Update progress
            epoch_loss += loss.item()
            batches_progress.desc = "Batches (Prev.loss: {:.4f}, Avg: {:.4f})".format(loss.item(), epoch_loss/max(1,(i)))
            batches_progress.update()

            # Delete tensors after use to free memory
            ''' Do not delete tensors before the last step (unless absolutely sure), as intermediate tensor's values
                might depend upon previous tensors, and they might get initialized if deleted,
                giving wrong results. So be very careful when deleting tensors. '''
            del output
            del target_seq
            del input_seq
            # del seq_len
            del loss
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        loss_plot.update(epoch_loss/(i+1), line_id = 0,update_time=False, display=False)

        # Epoch over
        print("Training loss: ", epoch_loss/(i+1), end=", ")
        epoch_progress.update()

        avg_loss += epoch_loss/(i+1)
        if epoch%1 == 0:
            # validate(on_training_set=True)
            acc = validate(model, test_dataloader, verbose=verbose) * 100
            train_accuracy.append(acc)
            loss_plot.update(float(acc/100), line_id = 1, display=True)

            # Export model if it has better accuracy than previous ones
            if acc >= cur_max_acc:
                cur_max_acc = acc
            if acc >= MAX_ACC:
                # Current model is better than previous ones
                MAX_ACC = acc
                print("\033[93mSaving current model with accuracy {:.4f} to disk...\033[0m".format(acc))
                f = open(savefile_name + ".dat","wb")
                torch.save(model.state_dict(), f)
                f.close()
                # Save dataloader to resume training if interrupted
                f = open(savefile_name + "_train_dataset.dat","wb")
                torch.save(train_dataloader, f)
                f.close()
                f = open(savefile_name + "_test_dataset.dat","wb")
                torch.save(test_dataloader, f)
                f.close()


    manager.stop()

def load_googlenet():
    from googlenet_pytorch import GoogLeNet 

    googlenet = GoogLeNet.from_pretrained('googlenet', 2)
    googlenet.aux_logits = False
    googlenet.to(device)
    print('done')
    return googlenet

from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler
from copy import deepcopy
def train_kfold(model, train_dataset, valid_dataset, test_dataset=None, batch_size=16, n_epochs=1, k_folds=5, optimizer_fn=None, lr=1e-4, criterion=None, MAX_ACC=60, savefile_name="model", verbose=False, use_augmentation = False):
    '''
    model: The required model to train (should take only the input sequence as input)
    train_dataloader: PyTorch Dataloader for the training set
    test_dataloader: PyTorch Dataloader for the testing set
    n_epochs: Number of epochs to run
    optimizer: Optimizer to use (object)
    criterion: Loss criterion to use (object)
    MAX_ACC: Current max accuracy of the model on disk (Any instance of training model with accuracy greater than MAX_ACC will overwrite the model on disk)
    savefile_name: Filename for the max accuracy model to be saved on disk (Do not give file extension. Eg: savefile_name=model1)
    verbose: Whether to display additional statistiics, including tp,tn,fp,fn, f-score, and misclassification margins
    '''
    manager = enlighten.get_manager()
    status_bar = manager.status_bar('Training...', color='black_on_peachpuff2', justify=enlighten.Justify.CENTER)
    status_bar2 = manager.status_bar('Waiting...', color='black_on_peachpuff2', justify=enlighten.Justify.CENTER)
    fold_progress = manager.counter(total=k_folds, desc="Folds", unit="folds", color="cyan")
    epoch_progress = manager.counter(total=n_epochs, desc="Epochs", unit="epochs", color="bright_red")
    batches_progress = manager.counter(total=len(train_dataset), desc="Batches", unit="batches", color="red")
    global validation_progress
    validation_progress = manager.counter(total=len(train_dataset), desc="Validating...", unit="batches", color="green", leave=False)

    loss_plot = plot_line(3)
    avg_loss = 0
    acc = 0
    val_acc = 0
    cur_max_acc = 0
    train_accuracy = []

    torch.manual_seed(42)

    if use_augmentation:                            # Do not use this method!
        train_dataset.augmentation()
        valid_dataset.augmentation()
    dataset = ConcatDataset([train_dataset, valid_dataset])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset)):
        epoch_progress.count = 0
        epoch_progress.start = time.time()
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valid_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=valid_subsampler)
        batches_progress.total = len(train_dataloader)
        validation_progress.total = len(valid_dataloader)

        t_model = deepcopy(model)
        t_model.train()

        optimizer = optimizer_fn(t_model, 1e-3)
        # model = deepcopy(t_model)
        # model.load_state_dict(original_state_dict)

        for epoch in range(1, n_epochs + 1):
            # with tqdm(total=len(dataloader), desc="Training...") as progress_bar:
            batches_progress.count = 0
            batches_progress.start = time.time()
            # batches_progress.clear()

            epoch_loss = 0
            for i, video in enumerate(train_dataloader):
                optimizer.zero_grad() # Clear existing gradients

                # Get training batch from dataloader
                input_seq, target_seq = video
                # input_seq, target_seq, seq_len = video

                status_bar.update("Training with:" + str([INV_LABELS[t.item()][0] for t in target_seq]))
                status_bar2.update("Accuracy: {:.4f},Avg: {:.4f}, Max: {:.4f}".format(val_acc, acc, cur_max_acc) + ", Loss: {:.4f}".format(avg_loss/max(1,epoch-1)))

                # Prepare data to be passed to googlenet CNN to extract features
                # googlenet_input_seq = googlenet_input_seq.to(device)
                # seq_len = seq_len.to(device)
                # input_seq = googlenet_model(googlenet_input_seq.cuda(), seq_len.cuda()).squeeze(0)    # Ouptput from googlenet => input to LSTM

                # Prepare input for the model
                input_seq = input_seq.to(device, dtype=torch.float32)


                # Get predictions
                output = t_model(input_seq)#.logits
                # output = model(input_seq, seq_len.to(device))#.logits

                # Prepare target sequence on required device to calculate loss
                target_seq = target_seq.to(device, dtype=torch.long)
                loss = criterion(output, target_seq.view(-1).long())

                # Update weights
                loss.backward() # Backpropagation and gradient calculation
                optimizer.step() # Weight updation

                # Update progress
                epoch_loss += loss.item()
                batches_progress.desc = "Batches (Prev.loss: {:.4f}, Avg: {:.4f})".format(loss.item(), epoch_loss/max(1,(i)))
                batches_progress.update()

                # Delete tensors after use to free memory
                ''' Do not delete tensors before the last step (unless absolutely sure), as intermediate tensor's values
                    might depend upon previous tensors, and they might get initialized if deleted,
                    giving wrong results. So be very careful when deleting tensors. '''
                del output
                del target_seq
                del input_seq
                # del seq_len
                del loss
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            loss_plot.update(epoch_loss/(i+1), line_id = 0,update_time=False, display=False)

            # Epoch over
            print("Training loss: ", epoch_loss/(i+1), end=", ")
            epoch_progress.update()

            avg_loss += epoch_loss/(i+1)
            if epoch%1 == 0:
                # Check accuracy after every 2 epochs
                # validate(on_training_set=True)
                val_acc = validate(t_model, valid_dataloader, verbose=verbose) * 100
                loss_plot.update(float(val_acc/100), line_id = 1, display=False)
                train_accuracy.append(val_acc)
                acc = np.sum(train_accuracy)/len(train_accuracy)
                print("-------- TEST SET -------")
                new_acc = validate(t_model, test_dataloader, verbose=verbose) * 100
                loss_plot.update(float(new_acc/100), line_id = 2, display=True)

                # Export model if it has better accuracy than previous ones
                if val_acc >= cur_max_acc:
                    cur_max_acc = val_acc
                if val_acc >= MAX_ACC:
                    # Current model is better than previous ones
                    MAX_ACC = val_acc
                    print("\033[93mSaving current model with accuracy {:.4f} to disk...\033[0m".format(val_acc))
                    f = open(savefile_name + ".dat","wb")
                    torch.save(t_model, f)
                    f.close()
                    # Save dataloader to resume training if interrupted
                    f = open(savefile_name + "_train_dataset.dat","wb")
                    torch.save(train_dataloader.dataset, f)
                    f.close()
                    f = open(savefile_name + "_valid_dataset.dat","wb")
                    torch.save(valid_dataloader.dataset, f)
                    f.close()
                    f = open(savefile_name + "_test_dataset.dat","wb")
                    torch.save(test_dataset, f)
                    f.close()

        fold_progress.update()
    manager.stop()