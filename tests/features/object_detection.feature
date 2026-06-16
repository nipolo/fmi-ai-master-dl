Feature: Detecting objects in an uploaded photo
  As a user of the object-detection web app
  I want to upload a photo and choose a confidence threshold
  So that I can see which everyday objects the model finds

  Scenario Outline: The confidence threshold controls how many objects are detected
    Given an image with a dog at 0.95 and a person at 0.80 confidence
    When I run detection with a confidence threshold of <threshold>
    Then <count> objects are detected
    And the most confident detection is a "<top_label>"

    Examples:
      | threshold | count | top_label |
      | 0.50      | 2     | dog       |
      | 0.90      | 1     | dog       |

  Scenario: No objects pass a very high threshold
    Given an image with a dog at 0.95 and a person at 0.80 confidence
    When I run detection with a confidence threshold of 0.99
    Then 0 objects are detected
