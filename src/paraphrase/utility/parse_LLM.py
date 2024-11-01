"""
    parse in context learning responses

"""
import string
import re

QUOTE_GUEST_IDENTIFIER = "Verbatim Quote Guest"
QUOTE_HOST_IDENTIFIER = "Verbatim Quote Host"


def extract_classification(text):
    return _is_yes(_extract_identifier_text(text))


def extract_host_quote(text, host_tokens, identifier=QUOTE_HOST_IDENTIFIER):
    host_quote = _extract_identifier_text(text, identifier=identifier)

    if _is_none(host_quote):
        raise ValueError(f"One of the host or guest quote is none: Host: {host_quote}")

    try:
        hl_list = _extract_quoted_phrase_indices(" ".join(host_tokens), host_quote)
    except ValueError as e:
        if "One of the host or guest quote is none" in str(e) or "Empty phrase found." in str(e):
            raise ValueError(e)
        else:
            raise ValueError(f"The _model hallucinated a quote for the host: {e}")
    return hl_list


def extract_guest_quote(text, guest_tokens, identifier=QUOTE_GUEST_IDENTIFIER):
    guest_quote = _extract_identifier_text(text, identifier=identifier)

    if _is_none(guest_quote):
        raise ValueError(f"One of the host or guest quote is none: Guest: {guest_quote}")

    try:
        gl_list = _extract_quoted_phrase_indices(" ".join(guest_tokens), guest_quote)
    except ValueError as e:
        if "One of the host or guest quote is none" in str(e) or "Empty phrase found." in str(e):
            raise ValueError(e)
        raise ValueError(f"The _model hallucinated a quote for the guest: {e}")
    return gl_list


def _extract_quoted_phrase_indices(original_text, quoting_text):
    """generated with GPT4
    Create a list indicating with 1 if a word in the original text is quoted, and 0 otherwise.
    This version allows for partial matches at the end of phrases, especially regarding punctuation.
    Raise an error if a quoted phrase is not found in the original text.

    lowercases and removes punctuation from the original text and quoting text.
        --> is not able to differentiate between "Hello, David." and "hello David."

    :param original_text: String in which the search is performed.
    :param quoting_text: String containing the quoted phrases.
    :return: List of 0s and 1s indicating whether each word in the original text is quoted.
    """

    if len(original_text) == 0:
        raise ValueError("Empty original text.")
    quoted_phrases = _extract_quoted_phrases(quoting_text, original_text)

    # Splitting the original text into words, keeping punctuation
    words = original_text.lower().split()  # re.sub(r'[^\w\s]', '', original_text).lower(), words = original_text.split()

    # Initializing the binary list with all zeros
    binary_list = [0] * len(words)

    # Updating the binary list for quoted words
    for phrase in quoted_phrases:
        phrase_found = False
        phrase_words = phrase.split()
        phrase_length = len(phrase_words)
        if phrase_length == 0:
            raise ValueError("Empty phrase found.")

        for i in range(len(words) - phrase_length + 1):
            # Comparing phrases while allowing for partial matches at the end
            if ' '.join(words[i:i + phrase_length - 1]) == ' '.join(phrase_words[:-1]):
                # Handling the last word separately to allow for punctuation differences
                try:
                    if words[i + phrase_length - 1].startswith(phrase_words[-1]):
                        for j in range(i, i + phrase_length):
                            binary_list[j] = 1
                        phrase_found = True
                        break  # Assuming each phrase appears only once
                except IndexError:
                    print("Index Error")
                    continue

        if not phrase_found:
            raise ValueError(f"Quoted phrase not found: '{phrase}' in '{original_text}'.")

    return binary_list


def _extract_quoted_phrases(quoting_text, original_text="", allow_without_marks=True):
    # Extracting phrases within quotes
    quoted_phrases = re.findall(r'"([^"]*)"', quoting_text)
    if len(quoted_phrases) == 0:
        # No matching quotation marks found.
        #   1. resolving strategy: might be quoting with <quote> <\quote>
        quoted_phrases = re.findall(r'<quote>(.*?)<\/quote>', quoting_text)

        if len(quoted_phrases) == 0:
            #   2. resolving strategy: try different quote, i.e., ' but only if original string does not include '
            if original_text.find("'") == -1:
                quoted_phrases = re.findall(r"'([^']*)'", quoting_text)

            if (len(quoted_phrases) == 0) and allow_without_marks:
                #   3. resolving strategy: might be quoting without quotation marks or
                #       just with a leading quotation mark
                #   remove leading quotation mark if it is there
                if quoting_text.startswith('"'):
                    quoting_text = quoting_text[1:]
                quoted_phrases = [quoting_text]
    # quoted_phrases = [re.sub(r'[^\w\s]', '', phrase).lower() for phrase in quoted_phrases]
    quoted_phrases = [phrase.lower() for phrase in quoted_phrases]
    return quoted_phrases


def get_indices_of_quoted_words(original_string, quoted_string):
    # Split the strings into lists of words
    original_words = original_string.split()
    quoted_words = quoted_string.split()

    # Initialize an empty list to store the indices
    indices = []

    # Iterate over the quoted words
    for word in quoted_words:
        # Find the index of the word in the original string
        index = original_words.index(word)
        # Append the index to the list of indices
        indices.append(index)

    return indices


def _extract_identifier_text(text, identifier="Classification"):
    """
        Assumes a form like
            "identifier: BLABLABLA\n"
        and returns
            "BLABLABLA"
        ATTENTION: this only returns the text after the FIRST occurrence of the identifier
    :param text:
    :return:
    """
    # Split the text by "Classification:"
    parts = text.split(f"{identifier}:", 1)

    # If "Classification:" was found, split the second part by linebreak and return the first piece
    if len(parts) > 1:
        return parts[1].split("\n", 1)[0].strip()

    # If "Classification:" was not found, raise error   # return an empty string or None
    raise ValueError(f"Identifier '{identifier}' not found.")
    # return ""


def _is_yes(input_string):
    # Define the variations of "Yes" and "No"
    variation_yes = "yes"
    variation_no = "no"

    # Remove leading and trailing whitespace from the input string, convert it to lowercase, and remove punctuation
    input_string = input_string.strip().lower().translate(str.maketrans('', '', string.punctuation))

    # Check if the input string starts with the variation
    if input_string.startswith(variation_yes):
        return True
    elif input_string.startswith(variation_no):
        return False
    else:
        raise ValueError("Input string does not start with 'Yes' or 'No'.")


def _is_none(input_string):
    """
        did the _model return "none" for the quote?
    :param input_string:
    :return:
    """

    # make sure that is not just none appearing in a quote
    quotes = _extract_quoted_phrases(input_string, allow_without_marks=False)
    if any("none" in quote.lower() for quote in quotes):
        return False
    else:
        # Remove punctuation from the text
        input_string = re.sub(r'[^\w\s]', '', input_string)

        # Convert the text to lower case
        input_string = input_string.lower()

        # Check if 'none' is in the text
        return 'none' in input_string.split()


def extract_is_none(text):
    """
        Also count as None if the identifiers are not present
    :param text:
    :return:
    """
    try:
        host_quote = _extract_identifier_text(text, identifier=QUOTE_HOST_IDENTIFIER)
        guest_quote = _extract_identifier_text(text, identifier=QUOTE_GUEST_IDENTIFIER)
    except ValueError:  # identifier not found
        return True
    try:
        none_host = _is_none(host_quote)
        none_guest = _is_none(guest_quote)
    except ValueError:
        raise ValueError("The _model hallucinated a quote for the host or guest.")
    if none_host and none_guest:  # both guest/host are annotated as None
        return True
    elif none_host or none_guest:
        raise ValueError("Only one of the host or guest is none.")
    else:
        return False
