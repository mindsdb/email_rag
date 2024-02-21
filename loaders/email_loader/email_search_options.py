from dataclasses import dataclass
import datetime


@dataclass
class EmailSearchOptions:
    '''Represents IMAP search options to use when searching emails'''
    # IMAP mailbox to search.
    mailbox: str
    # Search by email subject.
    subject: str
    # Search based on who the email was sent to.
    to_email: str
    # Search based on who the email was from.
    from_email: str
    # Search based on when the email was received.
    since_date: datetime.date
    until_date: datetime.date
    # Search for all emails after this ID.
    since_email_id: str
