# Streaming

## Producer
Producer generates random "product events", up to 5 per second, and 
sends them over a network connection.

## Consumer
There are 3 examples of stream consumers.
1. Simple
2. Consumer which makes some analytics
3. Consumer which update the state.

## Training
The streaming regression model provides two methods for usage:
- trainOn: This takes DStream[LabeledPoint] as its argument. This tells 
the model to train on every batch in the input DStream. It can be called 
multiple times to train on different streams.
- predictOn: This also takes DStream[LabeledPoint]. This tells the model 
to make predictions on the input DStream, returning a new DStream[Double] 
that contains the model predictions.

## Input data
