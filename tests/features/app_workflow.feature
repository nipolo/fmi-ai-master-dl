Feature: Driving the detection app
  As a user of the object-detection web app
  I want to pick a model, see a results table, and read a per-class summary
  So that I understand what the app detected and how

  Scenario: Choosing a model the app does not offer is rejected
    When I load a model called "totally-not-a-model"
    Then loading fails with an "unknown model" error

  Scenario: Detections are tabulated for display
    Given detections of a dog and a cat
    When I tabulate the detections
    Then the table has 2 rows
    And the table columns are "label, score, x1, y1, x2, y2"

  Scenario: Repeated classes are summarized per label
    Given detections of dog, dog, and person
    When I summarize the detections
    Then the summary reports 2 "dog" and 1 "person"

  Scenario: No row selection shows every detection on the photo
    Given detections of a dog and a cat
    When I select the table rows ""
    Then the shown detections are "dog, cat"

  Scenario: Selecting rows shows only those detections
    Given detections of a dog and a cat
    When I select the table rows "1"
    Then the shown detections are "cat"
