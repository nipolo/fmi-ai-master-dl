Feature: Detecting objects in an uploaded photo
  As a user of the object-detection web app
  I want to upload a photo and choose a confidence threshold
  So that I can see which everyday objects the model finds

  Scenario: Detecting objects above the confidence threshold
    Given an image with a dog at 0.95 and a person at 0.80 confidence
    When I run detection with a confidence threshold of 0.50
    Then 2 objects are detected
    And the most confident detection is a "dog"

  Scenario: Raising the threshold filters out weak detections
    Given an image with a dog at 0.95 and a person at 0.80 confidence
    When I run detection with a confidence threshold of 0.90
    Then 1 object is detected
    And the most confident detection is a "dog"

  Scenario: No objects pass a very high threshold
    Given an image with a dog at 0.95 and a person at 0.80 confidence
    When I run detection with a confidence threshold of 0.99
    Then 0 objects are detected
