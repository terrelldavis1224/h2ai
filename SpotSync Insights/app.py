
import flask
import base64
import d
import matplotlib
import random

matplotlib.use("Agg")

# Create the application.
APP = flask.Flask(__name__)


@APP.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return flask.render_template('index.html',random =random)




@APP.route('/dest', methods=['GET', 'POST'])
def dest():

    if flask.request.method == 'POST':
        g =flask.request.form.get('q')

        if g in d.genre_list:
            print ("okay")
            buf_list = d.get_genre_data_and_save_plots(g)
            image_data_list = [base64.b64encode(buf.getvalue()).decode('utf-8') for buf in buf_list]

            for buf in buf_list:
                buf.seek(0)
            g = "Music genre: "+g.capitalize() 
        else :
            buf_list = d.get_genre_data_and_save_plots('')
            image_data_list = [base64.b64encode(buf.getvalue()).decode('utf-8') for buf in buf_list]

            for buf in buf_list:
                buf.seek(0)
            g = 'All Music '

        return flask.render_template('dest.html', image_data_list=image_data_list, genre  = g, description=d.description)
    else :
        return flask.render_template('dest.html',search_suggestions = d.search_suggestions)


@APP.route('/temp',methods = ['POST','GET'])
def temp():

    if flask.request.method == 'POST':
        genre =flask.request.form.get('genre_form')
        if genre in d.genre_list:

            songdf=d.model_generation(genre)

            songdf=songdf[['track_name', 'name', 
            'track_popularity', 'uri',
            'album_name', 'album_popularity',
            'duration_sec', 'release_date', 'artist_popularity']]

            print(type(songdf))
            return flask.render_template('usemodel.html',search_suggestions = d.search_suggestions, songdf = songdf,cols=songdf.columns) 
        else:
            return flask.render_template('usemodel.html',search_suggestions = d.search_suggestions, songdf = None, warning = 1) 
    else:
        return flask.render_template('usemodel.html',search_suggestions = d.search_suggestions, songdf = None) 

@APP.route('/searchlike',methods = ['POST','GET'])
def searchlike():
    if flask.request.method == 'POST':
        song =flask.request.form.get('song_form')
        song_list_with_name = d.merged_spotify_data[d.merged_spotify_data['track_name'].str.contains(song)]
        if len(song_list_with_name) > 0:
            
            return flask.render_template('searchlike.html',search_suggestions = d.merged_spotify_data['track_name'],song_list_with_name=song_list_with_name)
        else:
            return  flask.render_template('searchlike.html',search_suggestions = d.merged_spotify_data['track_name'], warning = 1,song_list_with_name=None)

    return flask.render_template('searchlike.html',search_suggestions = d.merged_spotify_data['track_name'],song_list_with_name=None) 

@APP.route('/likesongs',methods = ['POST','GET'])
def likesongs():
    songid = flask.request.form.get('songid')
    closest_songs_df=d.closest_songs(songid)

    print(d.merged_spotify_data[d.merged_spotify_data['track_id'] == songid].iloc[0])
    return flask.render_template('likesongs.html',closest_songs_df=closest_songs_df,song =d.merged_spotify_data[d.merged_spotify_data['track_id'] == songid].iloc[0])



if __name__ == '__main__':
    APP.run(debug=True)