from utils import *
import librosa
from skimage import io
from torch import nn

from googlenet_pytorch import GoogLeNet
from torch.functional import F
class ensemble_model(nn.Module):
    def __init__(self, video_model, audio_model):
        super(ensemble_model, self).__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.fc = nn.Linear(4, 2)

    @torch.cuda.amp.autocast()
    def forward(self, video_input):
        (sequences, video_length, aud_op) = video_input
        vid_output = self.video_model((sequences.to(device), video_length.to(device)))
        aud_output = self.audio_model(aud_op.to(device))
        combined_output = torch.concat((vid_output,aud_output), 1)
        combined_output = F.relu(combined_output)
        output = self.fc(combined_output)
        return output


POOL_KERN = 4
POOL_STRIDE = 2
POOL_PAD = 1
FEATURE_DIM1 = int((1024 - POOL_KERN + (2 * POOL_PAD))/POOL_STRIDE + 1)
FEATURE_DIM2 = int((7 - POOL_KERN + (2 * POOL_PAD))/POOL_STRIDE + 1)
EMBED_DIMS = FEATURE_DIM1 * FEATURE_DIM2 * FEATURE_DIM2
print(EMBED_DIMS)


from torch.functional import F
class video_model(nn.Module):
    def __init__(self, pool_kern=POOL_KERN, pool_stride = POOL_STRIDE, pool_pad = POOL_PAD, CNN_embed_dim=EMBED_DIMS, h_RNN_layers=1, h_RNN=128, h_FC_dim=128, drop_p=0.3, num_classes=2):
        super(video_model, self).__init__()
        self.googlenet = GoogLeNet.from_pretrained('googlenet', CNN_embed_dim)
        self.googlenet.aux_logits = False

        self.pool_kern = pool_kern
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad

        self.pool = nn.MaxPool3d(self.pool_kern, self.pool_stride, self.pool_pad)
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        hidden_size = self.h_RNN
        self.h_RNN *= 2

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=hidden_size,
            num_layers=h_RNN_layers,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional = True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        # self.fc2 = nn.Linear(self.h_FC_dim, 1)
        self.sigmoid = nn.Sigmoid()
    @torch.cuda.amp.autocast()
    def forward(self, x_RNN):
        # x_RNN -> (vid, vid_len, aud)
        x_lengths = x_RNN[1]
        x_RNN = x_RNN[0]
        x_RNN = x_RNN.to(device)
        x_lengths = x_lengths.to(device)
        batch_id, timeframe, col_channels, img_x, img_y = x_RNN.size()
        # Pass through CNN to get features first
        batch_embed_seq = []
        for idx, vid in enumerate(x_RNN):
            t = x_lengths[idx].item()
            with torch.no_grad():
                cnn_embed_seq = self.googlenet(vid[:t])
                p4d = (0,0,0,timeframe-t)
                # print(cnn_embed_seq.size())
                cnn_embed_seq = F.pad(cnn_embed_seq, p4d, "constant", 0)
                batch_embed_seq.append(cnn_embed_seq)
                del cnn_embed_seq
                torch.cuda.empty_cache()
        batch_embed_seq = torch.stack(batch_embed_seq)

        x_lengths[x_lengths > timeframe] = timeframe
        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        # use input of descending length
        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(batch_embed_seq[perm_idx], lengths_ordered.cpu(), batch_first=True)
        self.LSTM.flatten_parameters()
        packed_RNN_out, (h_n_sorted, h_c_sorted) = self.LSTM(packed_x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        RNN_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_RNN_out, batch_first=True)
        RNN_out = RNN_out.contiguous()
        # RNN_out = RNN_out.view(-1, RNN_out.size(2))

        # reverse back to original sequence order
        _, unperm_idx = perm_idx.sort(0)
        RNN_out = RNN_out[unperm_idx]

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

vid_model = video_model(h_RNN_layers=3, h_RNN=256, h_FC_dim=256, drop_p=0.2)

from googlenet_pytorch import GoogLeNet
audio_model = GoogLeNet.from_pretrained('googlenet', 2)
audio_model.aux_logits = False
# audio_model = torch.load(open("googlenet_audio_1.3.dat", "rb"))
# for param in audio_model.parameters():
#     param.requires_grad = False

model = ensemble_model(vid_model, audio_model)
model.to(device)

FILENAME = "test"
model.load_state_dict(torch.load(open(FILENAME + ".dat", "rb"), map_location = torch.device(device)))
# train_dataloader = torch.load(open(FILENAME + "_train_dataset.dat", "rb"))
# test_dataloader = torch.load(open(FILENAME + "_test_dataset.dat", "rb"))
print("\033[93mLoaded pretrained model...\n\033[0m")

def predict(path):
    # Preprocess video
    import os
    os.system('rm -r temp_current_item_processing_001')
    os.system('mkdir temp_current_item_processing_001')

    DATASET_WRITE_DIR = 'temp_current_item_processing_001'
    labels = []
    sequences = []

    image_list = extractImages(path)
    cropped_images = get_sequences(image_list)
    # for idx_img, image in enumerate(cropped_images):
    #     # cv2.imwrite(DATASET_WRITE_DIR + '/' + str(idx_img) + '.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     cropped_images[idx_img] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Write back MFCC
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    sys.stdout = devnull
    clip = mp.VideoFileClip(path)
    clip.audio.write_audiofile(DATASET_WRITE_DIR + '/audio.wav')
    sys.stdout = old_stdout
    devnull.close()

    # Create input sequence
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # Image
    vid_sequences = []
    for idx, frame in enumerate(cropped_images):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        vid_sequences.append(frame)
    vid_sequences = torch.stack(vid_sequences)

    # Audio
    y, sr = librosa.load(DATASET_WRITE_DIR + '/audio.wav', sr = None)

    # MFCC Parameters
    n_fft=2048
    hop_length=512
    n_mels=128
    fmin=20
    fmax=8300
    top_db=80
    # Generate MFCC
    if y.shape[0]<5*sr:
        y = np.pad(y,int(np.ceil((5*sr-y.shape[0])/2)),mode='reflect')
    else:
        y=y[:5*sr]
    spec=librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)

    # For GoogleNet
    mels = np.log(spec + 1e-9) # add small number to avoid log(0)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    # save as PNG
    io.imsave('temp.png', img)
    # skimage.
    # return img
    from PIL import Image
    im = Image.open('temp.png')
    im = im.convert(mode='RGB')

    input_tensor = transform(im)
    aud_op = input_tensor

    # Load model and predict
    model.eval()
    # dataset_obj = dataset([DATASET_WRITE_DIR], [1], None, True)
    # input_seq, target_seq = dataset_obj.__getitem__(0)
    with torch.no_grad():
        # x_RNN -> (vid, vid_len, aud)
        vid_len = torch.tensor(len(vid_sequences), dtype=torch.int32)
        vid_len = torch.stack([vid_len])
        output = model((torch.stack([vid_sequences]), vid_len, torch.stack([aud_op])))
    output = output[0]
    if(output[0] > output[1]):       # Truth
        return 1
    elif(output[0] < output[1]):
        return 0
