class Story:
    def __init__(self, storyid, full_story, source, original_sentences,
                 original_title, parsed_sentences, parsed_title):
        self.storyid = storyid
        self.full_story = full_story
        self.source = source
        self.original_sentences = original_sentences
        self.original_title = original_title
        self.dparsed_sentences = parsed_sentences
        self.dparsed_title = parsed_title

