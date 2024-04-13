import numpy as np 
import pandas  as pd 
import matplotlib.pyplot as plt
import base64
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances



# Neural Network Model
class NNModel(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size_1, hidden_layer_size_2, output_size=1):
        super(NNModel, self).__init__()
        self.input_layer = nn.Linear(in_features=input_layer_size, out_features=hidden_layer_size_1)
        self.hidden_layer_1 = nn.Linear(in_features=hidden_layer_size_1, out_features=hidden_layer_size_2)
        self.hidden_layer_2 = nn.ReLU()
        self.output_layer = nn.Linear(in_features=hidden_layer_size_2, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = self.hidden_layer_2(x)
        output = self.output_layer(x)
        
        return output
    





spotify_artistdf=pd.read_csv("SpotSync Insights/csvs/spotify_artist_data_2023.csv")
spotify_data_df=pd.read_csv("SpotSync Insights/csvs/spotify_data_12_20_2023.csv")
spotify_features_data_df=pd.read_csv("SpotSync Insights/csvs/spotify_features_data_2023.csv")
spotify_tracks_data_df=pd.read_csv("SpotSync Insights/csvs/spotify_tracks_data_2023.csv")
spotify_albums_data_df = pd.read_csv("SpotSync Insights/csvs/spotify-albums_data_2023.csv")

merged_spotify_data=spotify_features_data_df.merge(spotify_tracks_data_df,how='inner',left_on="id",right_on ="id")
merged_spotify_data=merged_spotify_data.merge(spotify_data_df,how='inner',left_on="id",right_on ="track_id")
merged_spotify_data

genre_list = pd.concat([merged_spotify_data["genre_0"],merged_spotify_data["genre_1"]
          ,merged_spotify_data["genre_2"],merged_spotify_data["genre_3"],merged_spotify_data["genre_4"]]).unique()

columns_to_drop = ['id']

merged_spotify_data = merged_spotify_data.drop(columns=merged_spotify_data.filter(like='_y').columns)
merged_spotify_data = merged_spotify_data.drop(columns=columns_to_drop)

merged_spotify_data = merged_spotify_data.rename(columns=lambda x: x.replace('_x', ''))
merged_spotify_data = merged_spotify_data.dropna(subset=['track_name'])

metrics_col =['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','time_signature', 'duration_sec']


def closest_songs(track_id):
        spotify_data = merged_spotify_data[metrics_col]
        chosen_song = spotify_data.loc[merged_spotify_data['track_id'] == track_id].iloc[0]

        chosen_song_metrics = chosen_song[metrics_col].values.reshape(1, -1)

        all_songs_metrics = spotify_data[metrics_col].values

        euclidean_dist = euclidean_distances(chosen_song_metrics, all_songs_metrics)
        sorted_indices = euclidean_dist.argsort(axis=1)[0][1:11]
    
        selected_rows = merged_spotify_data.iloc[sorted_indices]
        selected_rows['euclidean_dist'] = euclidean_dist[0][sorted_indices]
        selected_rows['uri'] = selected_rows['uri'].str.replace('spotify:track:', 'https://open.spotify.com/track/')

        return selected_rows

def model_generation(genre):

    
    spotify_data=merged_spotify_data[(merged_spotify_data['genre_0']== genre ) |
                                              (merged_spotify_data['genre_1']== genre ) |
                                              ( merged_spotify_data['genre_2']== genre ) |
                                               (merged_spotify_data['genre_3']== genre ) ]
    Y = spotify_data['track_popularity']
    X = spotify_data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'duration_ms', 'time_signature', 'duration_sec', 'track_number']]


    X=(X-X.mean())/X.std()
    Y=(Y-Y.mean())/Y.std()


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    X_train = torch.FloatTensor(X_train.values)
    X_test = torch.FloatTensor(X_test.values)
    Y_train = torch.FloatTensor(Y_train.values)
    Y_test = torch.FloatTensor(Y_test.values)

    model = NNModel(input_layer_size=len(X.columns), hidden_layer_size_1=16, hidden_layer_size_2=8, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train 
    epochs = 50
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, Y_train)
    
        losses.append(loss.item())
    
    # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    data =torch.FloatTensor(X.values)
    l =[]
    
    with torch.no_grad():
        for i,d in enumerate(data):
            pred= model.forward(d)
            l.append( pred.item())  

    spotify_data['pred_pop'] = l

    spotify_data['norm_pop'] = Y  
    spotify_data['uri'] = spotify_data['uri'].str.replace('spotify:track:', 'https://open.spotify.com/track/')

    return spotify_data







def get_genre_data_and_save_plots(genre):
    list_of_int_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                        'duration_ms', 'time_signature',
                        'duration_sec', 'total_tracks']
    
   
    all_spotify_data= merged_spotify_data

    if  genre != "" :
        all_spotify_data = all_spotify_data[(all_spotify_data['genre_0']== genre ) |
                                              (all_spotify_data['genre_1']== genre ) |
                                              ( all_spotify_data['genre_2']== genre ) |
                                               (all_spotify_data['genre_3']== genre ) ]

    
    top_song_spotify_data=(all_spotify_data[all_spotify_data['track_popularity'] > 50])
    bottom_song_spotify_data=(all_spotify_data[all_spotify_data['track_popularity'] < 50])
    x_positions = np.array([0, 0.2, 0.4])
    list_of_int_cols = list_of_int_cols[:-3]
    
    buf_list = []  # List to store image buffers

    for column in list_of_int_cols:
        top = top_song_spotify_data[column].mean()
        all_data = all_spotify_data[column].mean()
        bottom = bottom_song_spotify_data[column].mean()

        # Create a new figure for each iteration
        fig, ax = plt.subplots()
        ax.set_facecolor('grey')
        fig.set_facecolor('white')
        ax.bar(x_positions[0], height=top, width=0.1, label='Top Song', edgecolor="black", color=np.random.rand(1,3))
        ax.bar(x_positions[1], height=all_data, width=0.1, label='All Data', edgecolor="black", color=np.random.rand(1,3))
        ax.bar(x_positions[2], height=bottom, width=0.1, label='Bottom Song', edgecolor="black", color=np.random.rand(1,3))

        ax.set_xlabel('Categories')
        ax.set_ylabel('Mean Values')
        ax.set_title(f'Mean Values for {column}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['Top Song', 'All Data', 'Bottom Song'])
        ax.text(x_positions[0], top, '%.2f' % top, ha='center', va='bottom')
        ax.text(x_positions[1], all_data, '%.2f' % all_data, ha='center', va='bottom')
        ax.text(x_positions[2], bottom, '%.2f' % bottom, ha='center', va='bottom')

        # Save the figure to a buffer
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf_list.append(buf)


        

        # Close the figure to free up resources
        plt.close(fig)

    return buf_list


description = ['A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.',
            'Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.','The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.','The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.','Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.','Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.','The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.','Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.','Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.','A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).','The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.']






search_suggestions=['pop', 'rap', 'hip hop', 'pop rap', 'rock', 'dance pop', 'atl hip hop', 'k-pop', 'canadian pop', 'r&b', 'trap', 'permanent wave', 'alternative metal', 'modern rock', 'melodic rap', 'canadian hip hop', 'gangster rap', 'edm', 'alternative rock', 'k-pop boy group', 'classic rock', 'art pop', 'country', 'urbano latino', 'album rock', 'contemporary country', 'southern hip hop', 'latin pop', 'trap latino', 'urban contemporary', 'post-grunge', 'detroit hip hop', 'country road', 'k-pop girl group', 'uk pop', 'pov: indie', 'reggaeton', 'soul', 'chicago rap', 'pop dance', 'nu metal', 'east coast hip hop', 'hip pop', 'hard rock', 'canadian contemporary r&b', 'art rock', 'brostep', 'queens hip hop', 'adult standards', 'mellow gold', 'country dawn', 'conscious hip hop', 'bolero', 'big room', 'rap metal', 'funk metal', 'house', 'pittsburgh rap', 'indie pop', 'funk rock', 'new orleans rap', 'dfw rap', 'underground hip hop', 'pop punk', 'post-teen pop', 'contemporary r&b', 'melancholia', 'g funk', 'viral pop', 'electro', 'rage rap', 'lounge', 'modern alternative rock', 'grunge', 'atl trap', 'trap queen', 'sertanejo universitario', 'sertanejo', 'punk', 'barbadian pop', 'electro house', 'north carolina hip hop', 'oxford indie', 'hardcore hip hop', 'heartland rock', 'slap house', 'latin hip hop', 'west coast rap', 'dirty south rap', 'soundtrack', 'indietronica', 'k-rap', 'musica mexicana', 'classical', 'easy listening', 'swing', 'dutch edm', 'alternative r&b', 'psychedelic rock', 'bachata', 'filter house', 'vocal jazz', 'outlaw country', 'arrocha', 'agronejo', 'orchestral soundtrack', 'progressive electro house', 'dark trap', 'classic oklahoma country', 'talent show', 'modern country rock', 'philly rap', 'uk contemporary r&b', 'singer-songwriter', 'neo soul', 'folk', 'r&b en espanol', 'british soul', 'cloud rap', 'houston rap', 'latin alternative', 'progressive rock', 'chamber pop', 'rock en espanol', 'alt z', 'irish rock', 'cali rap', 'mandopop', 'pluggnb', 'classic country pop', 'sierreno', 'mexican pop', 'classic texas country', 'soft rock', 'alternative hip hop', 'metropopolis', 'boy band', 'indie rock', 'filmi', 'pop soul', 'singer-songwriter pop', 'metal', 'complextro', 'old school atlanta hip hop', 'modern bollywood', 'garage rock', 'norteno', 'corrido', 'metalcore', 'sad sierreno', 'new wave', 'ohio hip hop', 'motown', 'latin rock', 'jazz blues', 'jazz', 'country rock', 'folk rock', 'classic soul', 'argentine hip hop', 'modern country pop', 'afrobeats', 'trap argentino', 'banda', 'nigerian pop', 'sertanejo pop', 'piano rock', 'arkansas country', 'ranchera', 'bedroom pop', 'late romantic era', 'nyc rap', 'australian pop', 'taiwan pop', 'miami hip hop', 'pop rock', 'british invasion', 'canadian rock', 'electronica argentina', 'canadian metal', 'ccm', 'nu-metalcore', 'indie folk', 'pop electronico', 'corridos tumbados', 'blues rock', 'baton rouge rap', 'funk', 'sheffield indie', 'uk dance', 'uk hip hop', 'glam metal', 'glam rock', 'quiet storm', 'pop nacional', 'crunk', 'christian music', 'reggaeton colombiano', 'new wave pop', 'russian romanticism', 'madchester', 'stomp and holler', 'c-pop', 'baroque pop', 'german soundtrack', 'dance rock', 'zhongguo feng', 'classical drill', 'britpop', 'new romantic', 'worship', 'british folk', 'sped up', 'neo mellow', 'dancehall', 'la indie', 'golden age hip hop', 'neo-classical', 'drill', 'puerto rican pop', 'pop emo', 'neo-psychedelic', 'latin arena pop', 'indie poptimism', 'girl group', 'bow pop', 'brazilian hip hop', 'australian psych', 'kentucky indie', 'acid rock', 'kentucky roots', 'azontobeats', 'rock-and-roll', 'neon pop punk', 'disco', 'compositional ambient', 'tropical', 'tennessee hip hop', 'brooklyn indie', 'r&b argentino', 'lilith', 'electropop', 'broadway', 'birmingham metal', 'downtempo', 'german hip hop', 'escape room', 'british soundtrack', 'tropical house', 'yacht rock', 'uk post-punk', 'opm', 'plugg', 'christian alternative rock', 'folk-pop', 'nigerian hip hop', 'industrial metal', 'indie pop rap', 'industrial', 'industrial rock', 'modern indie pop', 'lgbtq+ hip hop', 'soul blues', 'blues', 'acoustic pop', 'dutch house', 'florida rap', 'emo rap', 'stomp and flutter', 'pop folk', 'reggae', 'celtic rock', 'uk alternative pop', 'drum and bass', 'rap rock', 'pop violin', 'electronica', 'detroit trap', 'electronic trap', 'german metal', 'candy pop', 'asian american hip hop', 'countrygaze', 'indie soul', 'ectofolk', 'oakland hip hop', 'white noise', 'old school hip hop', 'florida drill', 'latin christian', 'springfield mo indie', 'jazz trio', 'anime', 'cantautora mexicana', 'dream pop', 'mariachi', 'show tunes', 'spanish pop', 'folk punk', 'minneapolis sound', 'chicago drill', 'australian dance', 'new orleans jazz', 'reggaeton flow', 'boom bap brasileiro', 'world worship', 'synth funk', 'rockabilly', 'reggae fusion', 'canadian punk', 'salsa', 'electric blues', 'merseybeat', 'classical performance', 'jazz rap', 'tejano', 'memphis hip hop', 'grime', 'j-pop', 'weirdcore', 'athens indie', 'dancefloor dnb', 'celtic punk', 'kentucky hip hop', 'banda jalisciense', 'emo', 'tecnobanda', 'pop flamenco', 'alternative dance', 'screamo', 'bebop', 'classical era', 'pixie', 'colombian pop', 'salsa puertorriquena', 'russian hip hop', 'pop urbaine', 'color noise', 'europop', 'shoegaze', 'rain', 'torch song', 'modern dream pop', 'baltimore indie', 'seattle hip hop', 'korean r&b', 'eurodance', 'baroque', 'liverpool indie', 'musica sonorense', 'indian instrumental', 'ghanaian pop', 'new age piano', 'melodic metalcore', 'mexican rock', 'groove metal', 'detroit rock', 'russian trap', 'ambient', 'gothic metal', 'glitchcore', 'early music', 'american folk revival', 'neue deutsche harte', 'korean instrumental', 'neo-synthpop', 'dance-punk', 'scottish singer-songwriter', 'desi pop', 'chill r&b', 'sertanejo tradicional', 'chanson', 'hyperpop', 'meme rap', 'uk funky', 'viral rap', 'progressive house', 'early romantic era', 'rock cristiano', 'pinoy hip hop', 'new jack swing', 'german romanticism', 'italian pop', 'lo-fi', 'victoria bc indie', 'cello', 'philly soul', 'oakland indie', 'progressive metalcore', 'alternative emo', 'belgian edm', 'indie rock italiano', 'vapor soul', 'german cloud rap', 'tagalog rap', 'r&b francais', 'string duo', 'pop r&b', 'southern soul', 'elephant 6', 'dreamo', 'korean pop', 'progressive deathcore', 'uk metalcore', 'roots reggae', 'swedish pop', 'punk blues', 'instrumental lullaby', 'taiwan singer-songwriter', 'dark r&b', 'new jersey rap', 'canadian pop punk', 'german trap', 'beatlesque', 'pop romantico', 'r&b brasileiro', 'tollywood', 'drift phonk', 'pop rap brasileiro', 'melodic house', 'modern blues rock', 'cyberpunk', 'singeli', 'memphis soul', 'red dirt', 'afro r&b', 'harlem renaissance', 'electronic rock', 'venezuelan hip hop', 'jazz saxophone', 'afrofuturism', 'bedroom r&b', 'canadian indie', 'canadian ccm', 'country pop', 'vintage broadway', 'fictitious orchestra', 'bachata dominicana', 'canadian country', 'future house', 'piano cover', 'deep ccm', 'rap belge', 'old school thrash', 'gothic symphonic metal', 'swedish electropop', 'taiwan indie', 'classic opm', 'rap calme', 'movie tunes', 'philly indie', 'indie punk', 'canadian trap', 'art punk', 'trap venezolano', 'rap latina', 'bubblegum pop', 'bronx drill', 'early modern classical', 'punjabi pop', 'southern rock', 'cantopop', 'lo-fi emo', 'cool jazz', 'reggaeton cristiano', 'brooklyn drill', 'idol', 'wave', 'french pop', 'australian hip hop', 'deep disco house', 'mexican hip hop', 'modern indie folk', 'ska mexicano', 'trip hop', 'rap cristiano', 'death metal', 'vallenato', 'hindi hip hop', 'german power metal', 'lo-fi chill', 'midwest americana', 'stutter house', 'french synthpop', 'livetronica', 'british blues', 'otacore', 'ska punk', 'chicago indie', 'psychedelic hip hop', 'korean ost', 'neomelodici', 'polish classical', 'nyc pop', 'japanese classical', 'naija worship', 'austin hip hop', 'indie anthem-folk', 'hip house', 'boston hip hop', 'dublin indie', 'jazz trumpet', 'new york drill', 'comic', 'deep talent show', 'post-romantic era', 'south african rock', 'synthpop', 'dixieland', 'dutch metal', 'background piano', 'nz pop', 'lo-fi beats', 'sad lo-fi', 'symphonic metal', 'video game music', 'moombahton', 'japanese soundtrack', 'comedy rap', 'etherpop', 'italian baroque', 'french hip hop', 'german baroque', 'jamaican dancehall', 'alte', 'indonesian pop', 'anime score', 'gen z singer-songwriter', 'slowcore', 'french rock', 'israeli classical piano', 'nashville sound', 'deep house', 'chicago bop', 'big beat', 'latin viral pop', 'transpop', 'instrumental hip hop', 'rap dominicano', 'spanish rock', 'rebel blues', 'impressionism', 'jazztronica', 'hk-pop', 'gym phonk', 'experimental pop', 'singaporean pop', 'portland indie', 'jungle', 'musica popular colombiana', 'small room', 'jazz piano', 'singaporean mandopop', 'italo dance', 'utopian virtual', 'viking folk', 'christian indie', 'nu jazz', 'classic canadian rock', 'jazz pop', 'boom bap', 'buffalo hip hop', 'rune folk', 'rock andaluz', 'ragga jungle', 'contemporary post-bop', 'nursery', 'melodic drill', 'bakersfield sound', 'bedroom soul', 'nordic folk', 'hypnagogic pop', 'epicore', 'chicano rap', 'deep euro house', 'german dance', 'albuquerque indie', 'new americana', 'slowed and reverb', 'upstate ny rap', 'uk alternative hip hop', 'musica para ninos', 'german pop', 'alabama rap', 'dembow', 'french shoegaze', 'scam rap', 'lo-fi rap', 'classical piano', 'french indie pop', 'nashville indie', 'aesthetic rap', 'cancion melodica', "children's music", 'indonesian r&b', 'rock drums', 'urbano espanol', 'japanese metalcore', 'chill phonk', 'pixel', 'ghanaian hip hop', 'environmental', 'uzbek pop', 'british indie rock', 'post-disco', 'dubstep', 'westcoast flow', 'water', 'azonto', 'roots worship', 'south african alternative', 'cumbia', 'indonesian rock', 'norwegian pop', 'singaporean singer-songwriter', 'modern reggae', 'harlem hip hop', 'chinese classical performance', 'piano blues', 'reparto', 'african rock', 'balkan beats', 'funk pop', 'j-metal', 'chinese indie', 'new jersey underground rap', 'northern soul', 'japanese vgm', 'chinese r&b', 'trap boricua', 'hardcore techno', 'sleep', 'classic colombian pop', 'chill house', 'electro swing', 'anthem worship', 'breakbeat', 'intelligent dance music', 'german alternative rap', 'j-poprock', 'classic italian pop', 'chinese classical piano', 'bubblegum dance', 'glitchbreak', 'classic j-pop', 'cubaton', 'indie hip hop', 'modern power pop', 'funk carioca', 'belgian dance', 'sda a cappella', 'sigilkore', 'indonesian pop rock', 'estonian hip hop', 'indie r&b', 'disco house', 'techno remix', 'anime rock', 'german drill', 'indonesian alternative rock', 'south african hip hop', 'rave', 'scandipop', 'russian modern classical', 'orchestral performance', 'electro latino', 'tech house', 'japanese singer-songwriter', 'doo-wop', 'malaysian mandopop', 'rap conscient', 'sound', 'cantautor', 'hamburg hip hop', 'french indietronica', 'scenecore', 'spanish new wave', 'french romanticism', 'caucasian classical', 'acoustic rock', 'anti-folk', 'background music', 'cumbia chilena', 'cumbia villera', 'indie game soundtrack', 'russian pop', 'acoustic punk', 'pinoy r&b', 'variete francaise', 'deep underground hip hop', 'bassline', 'dancehall queen', 'musica bajacaliforniana', 'italian adult pop', 'classic cantopop', 'contemporary classical', 'focus', 'glitch hop', 'tamil pop', 'jazz funk', 'rap canario', 'pinoy rock', 'christian hip hop', "preschool children's music", 'funk rj', 'christmas instrumental', 'viral trap', 'melodic techno', 'pacific islands pop', 'grupera', 'rawstyle', 'hard bop', 'ambient lo-fi', 'witch house', 'minnesota hip hop', 'african-american classical', 'virginia hip hop', 'diy emo', 'rennes indie', 'bossa nova', 'modern alternative pop', 'hardcore punk', 'british country', 'trap triste', 'hexd', 'modern salsa', 'theme', 'charva', 'afroswing', "australian children's music", 'east coast reggae', 'sudanese hip hop', 'afropop', 'background jazz', 'shimmer pop', 'bubblegrunge', 'banjo', 'classical cello', 'classic bollywood', 'neo-singer-songwriter', 'dutch r&b', 'dutch trance', 'uk dnb', 'underground power pop', 'gym hardstyle', 'ska', 'amapiano', 'cumbia peruana', 'aussietronica', 'mainland chinese pop', 'hong kong rock', 'new beat', 'trova', 'chill abstract hip hop', 'speed metal', 'afrikaans', 'sudanese pop', 'violin', 'contemporary vocal jazz', 'uk experimental electronic', 'socal pop punk', 'musical advocacy', 'pinoy trap', 'hawaiian', 'sophisti-pop', 'nouvelle chanson francaise', 'experimental r&b', 'trance', 'bronx hip hop', 'handpan', 'thai viral pop', 'dutch hip hop', 'sufi', 'melbourne bounce', 'pop house', 'braindance', 'drain', 'uk drill', 'chinese singer-songwriter', 'auckland indie', 'classic pakistani pop', 'j-division', 'skate punk', 'brazilian edm', 'funk ostentacao', 'viral afropop', 'pinoy indie', 'chill pop', 'dembow dominicano', 'pop venezolano', 'swedish reggae', 'rap guarulhense', 'arab alternative', 'melodic death metal', 'jazz boom bap', 'fvnky rimex', 'chillwave', 'shamanic', 'minimal techno', 'trap maroc', 'proto-hyperpop', 'melbourne bounce international', 'melanesian pop', 'vietnamese melodic rap', 'polish hip hop', 'j-rock', 'power pop', 'gothic rock', 'ambient idm', 'monterrey indie', 'alternative country', 'mollywood', 'afro house', 'nu-cumbia', 'jazz fusion', 'grimewave', 'scottish rock', 'progressive bluegrass', 'german techno', 'nottingham hip hop', 'phonk brasileiro', 'rock kapak', 'social media pop', 'tanzanian pop', 'north east england indie', 'irish pop', 'drone', 'british singer-songwriter', 'mantra', 'zolo', 'salsa peruana', 'cosmic american', 'swedish indie pop', 'experimental vocal', 'korean singer-songwriter', 'indiana hip hop', 'spiritual hip hop', 'french tech house', 'swiss house', 'novelty', 'modern folk rock', 'belly dance', 'victorian britain', 'cook islands pop', 'ukrainian indie', 'honky tonk', 'drumless hip hop', 'ukrainian classical', 'norwegian indie', 'xtra raw', 'qawwali', 'danish electronic', 'trancecore', 'japanese viral pop', 'protopunk', 'pagode baiano', 'neoclassical darkwave', 'solo wave', 'japanese electropop', 'new rave', 'chicago punk', 'drill and bass', 'danish indie', 'english baroque', 'swedish trap pop', 'traphall', 'japanese soul', 'desi hip hop', 'jain bhajan', 'lebanese pop', 'binaural', 'roots rock', 'phonk', 'belgian dnb', 'bass house', 'electra', 'sad rap', 'german trance', "women's music", 'psychedelic soul', 'nightcore', 'rap metalcore', 'derby indie', 'kosovan pop', 'v-pop', 'fantasy metal', 'egyptian trap', 'country rap', 'classic city pop', 'soul jazz', 'classic venezuelan pop', 'comedienne', 'danish indie pop', 'italian hip hop', 'gujarati pop', 'chill breakcore', 'albanian hip hop', 'deep tropical house', 'egyptian hip hop', 'battle rap', 'cuarteto', 'danish techno', 'gambian hip hop', 'k-indie', 'classify', 'french techno', 'classic house', 'british comedy', 'musica chihuahuense', 'eau claire indie', 'afghan pop', 'reggaeton mexicano', 'latin viral rap', 'sgija', 'alternative pop', 'finnish classical', 'breakcore', 'world devotional', 'instrumental grime', 'indiecoustica', 'yodeling', 'austin americana', 'early us punk', 'classic garage rock', 'gothenburg metal', 'nantes indie', 'albany ny indie', 'chinese hip hop', 'chinese indie pop', 'vietnamese hip hop', 'funk 150 bpm', 'new age', 'fijian pop', 'finnish metal', 'norwegian jazz', 'irish singer-songwriter', 'scottish new wave', 'guaracha', 'noise pop', 'canadian contemporary country', 'rap kreyol', 'australian indie folk', 'pop argentino', 'speedrun', 'canadian electropop', 'bolobedu house', 'drill francais', 'japanese chillhop', 'jovem guarda', 'frauenrap', 'lo-fi jazzhop', 'bongo flava', 'chill guitar', 'hyperpop brasileiro', 'lo-fi product', 'reggaeton chileno', 'pop reggaeton', 'classic girl group', 'japanese vtuber', 'black americana', 'rap df', 'gregorian chant', 'trap chileno', 'bhajan', 'modern j-rock', 'anime phonk', 'calming instrumental', 'bubblegum bass', 'brit funk', 'gainesville indie', 'brill building pop', 'rap underground espanol', 'trap brasileiro', 'hi-nrg', 'swedish gangsta rap', 'deep groove house', 'meme', 'irish hip hop', 'quebec indie', 'hip hop tuga', 'indie curitibano', 'musica neoleonesa', 'speed up brasileiro', 'punjabi lo-fi', 'russian classical piano', 'nashville hip hop', 'german post-hardcore', 'indie psych-pop', 'instrumental funk', 'hare krishna', 'bases de freestyle', 'jump blues', 'brazilian reggae', 'idol kayo', 'ragtime', 'folktronica', 'music hall', 'zouk', 'cumbia editada', 'pop edm', 'albanian pop', 'trap dominicano', 'russian emo rap', 'halifax indie', 'trap italiana', 'aussie drill', 'cartoon', 'arab pop', 'ye ye', 'spanish hip hop', 'jamaican hip hop', 'sacramento hip hop', 'horror synth', 'baltimore hip hop', 'traprun', 'bubble trance', 'trap tuga', 'bolero mexicano', 'shanty', 'aggressive phonk', 'south african house', 'mpb', 'swedish hip hop', 'tanzanian hip hop', 'uk dancehall', 'trap soul', 'plug brasileiro', 'atlanta bass', 'teen pop', 'urbano chileno', 'folklore ecuatoriano', 'mississippi hip hop', 'uk house', 'rap anime', "canzone d'autore", 'russian grime', 'ambient pop', 'chicago hardcore', 'latin worship', 'jam band', 'euphoric hardstyle', 'deep southern trap', 'apostolic worship', 'rhythm and blues', 'australian trap', 'cumbia sonidera', 'persian pop', 'miami bass', 'malaysian pop', 'south carolina hip hop', 'dream trance', 'trap carioca', 'indie electropop', 'neo r&b', 'early avant garde', 'new french touch', 'ocean', 'frenchcore', 'finnish metalcore', 'chicago soul', 'ritmo kombina', 'latin house', 'sichuanese hip hop', 'popwave', 'austindie', 'antiviral pop', 'bounce', 'popping', 'classic hardstyle', 'australian rock', 'brutal deathcore', 'queer country', 'peruvian indie', 'pink noise', 'deathcore', 'kumaoni pop', 'british modern classical', 'japanese teen pop', 'japanese alternative rock', 'garhwali pop', 'icelandic experimental']