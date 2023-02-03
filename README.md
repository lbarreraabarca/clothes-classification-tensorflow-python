# Classify images of clothing API


https://www.tensorflow.org/tutorials/keras/classification


```bash
curl --location --request POST 'http://localhost:8080/api/v1/product/classificator' \
--header 'Content-Type: application/json' \
--data-raw '{
    "url": "https://home.ripley.cl/store/Attachment/WOP/D143/2000378319764/2000378319764-1.jpg"
}'
```