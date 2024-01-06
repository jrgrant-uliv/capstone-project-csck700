import matplotlib.pyplot as plt
from wordcloud import WordCloud

file_name = "../../dataset/tbird/Thunderbird_5K.log"
# Read text data from a file
with open(file_name, 'r') as file:
    text_data = file.read()

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# plt.title("Word Cloud Example")
plt.show()
