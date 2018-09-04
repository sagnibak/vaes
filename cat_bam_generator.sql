--order all cat images by how much "catty" they are
CREATE TABLE cat_mid AS
  SELECT scores.mid
  FROM modules, scores
  WHERE modules.mid = scores.mid
  ORDER BY scores.content_cat DESC;

--get the top 50,000 vectorart images
CREATE TABLE vectorart50k AS
  SELECT scores.mid
  FROM scores
  ORDER BY scores.media_vectorart DESC
  LIMIT 50000;

--get the top 50,000 cat images
CREATE TABLE cat50k AS
  SELECT scores.mid
  FROM scores
  ORDER BY scores.content_cat DESC
  LIMIT 50000;

--get the top 5% vector art cats
CREATE TABLE vectorcat AS
  SELECT cat50k.mid
  FROM cat50k, vectorart50k
  WHERE cat50k.mid = vectorart50k.mid;

--get the top 5% water color cats
CREATE TABLE watercat AS
  SELECT cat50k.mid
  FROM cat50k, watercolor50k
  WHERE cat50K.mid = watercolor50k.mid;

SELECT src
  FROM cat_mid, modules
  WHERE cat_mid.mid = modules.mid LIMIT 10000;

.OUTPUT watercat.csv
SELECT src
  FROM watercat, modules
  WHERE watercat.mid = modules.mid;

.OUTPUT vectorcat.csv
SELECT src
  FROM vectorcat, modules
  WHERE vectorcat.mid = modules.mid;

select cat50k.mid from watercolor50k, cat50k where cat50k.mid=watercolor50k.mid;

select src from modules where modules.mid = 16238884;

select count(watercat.mid) from vectorcat, watercat where vectorcat.mid = watercat.mid;
--output: 0