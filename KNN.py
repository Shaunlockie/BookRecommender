import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tkinter import *
from tkinter import ttk

root = Tk()
root.title("Book Recommender")  
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')
books = books.drop(['goodreads_book_id','best_book_id','work_id','books_count','isbn13','language_code','work_ratings_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url','work_text_reviews_count'],axis=1)

#Merge then drop empty rows
combine_book_rating = pd.merge(ratings,books, on='book_id')
combine_book_rating = combine_book_rating.dropna(axis=0, subset = ['title'])

#find groupby then count book rating
book_rating_count = (combine_book_rating.
     groupby(by=['title'])['average_rating'].
     count().
     reset_index().
     rename(columns={'average_rating': 'total_rating_count'})
     [['title','total_rating_count']])
book_rating_count.head()

#merge books rating with total 
rating_total_rating_count = combine_book_rating.merge(book_rating_count,left_on = 'title', right_on='title', how='left')
rating_total_rating_count.head()

pd.set_option('display.float_format', lambda x: '%.3f' % x)

def Desc():
    print(book_rating_count['total_rating_count'].describe())
    print(book_rating_count['total_rating_count'].quantile(np.arange(.9,1,0.1)))

# create popularity threshold then remove any under that number that drop duplicates and create matrix
popularity_threshold = 100
rating_popular_book = rating_total_rating_count.query('total_rating_count >= @popularity_threshold')
rating_popular_book.head()
rating_popular_book = rating_popular_book.drop_duplicates(['user_id','title'])
rating_popular_book_pivot = rating_popular_book.pivot(index='title', columns='user_id',values='rating').fillna(0)
rating_popular_book_matrix = csr_matrix(rating_popular_book_pivot.values)  

#fit model
model_knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
model_knn.fit(rating_popular_book_matrix)
    
def clicked(event):
    myLabel = Label(root, text=combobox.current()).pack()
    
# create list of names and then populate combo box with names
options = rating_popular_book_pivot.index.tolist()
combobox = ttk.Combobox(root, value=options)
combobox.bind("<<ComboboxSelected>>", clicked)


def MainButton():
    #main button function to find recommendation
    query_index = combobox.current()
    
    distances, indices = model_knn.kneighbors(rating_popular_book_pivot.iloc[query_index, :].values.reshape(1,-1), n_neighbors=6)
    
    for i in range(0, len(distances.flatten())):
         if i == 0:
              recinput = "Recommendations for: {0}:\n".format(rating_popular_book_pivot.index[query_index])
              recinputLabel = Label(root, text=query_index)
              recinputLabel.pack()
         else:
             recOutput = "{0}: {1}, with distance of {2}".format(i, rating_popular_book_pivot.index[indices.flatten()[i]], distances.flatten()[i])
             recOutputLabel = Label(root, text=recOutput)
             recOutputLabel.pack()

def RatingDistrib():
    #displaying bar chart of rating distributions
    plt.rc("font", size=15)
    ratings.rating.value_counts(sort=False).plot(kind='bar')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('system1.png', bbox_inches='tight')
    plt.show()
    
myButton = Button(root, text="Recommend", command=MainButton)
combobox.pack()
myButton.pack()


Desc()
RatingDistrib()
root.mainloop()