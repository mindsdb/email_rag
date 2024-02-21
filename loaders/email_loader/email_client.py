from datetime import datetime, timedelta
import email
import imaplib

import pandas as pd

from loaders.email_loader.email_search_options import EmailSearchOptions


class EmailClient:
    '''Class for searching emails using IMAP (Internet Messaging Access Protocol)'''

    _DEFAULT_SINCE_DAYS = 10

    def __init__(
            self,
            email: str,
            password: str,
            imap_server: str = 'imap.gmail.com'):
        self.email = email
        self.password = password
        self.imap_server = imaplib.IMAP4_SSL(imap_server)

    def select_mailbox(self, mailbox: str = 'INBOX'):
        '''Logs in & selects a mailbox from IMAP server. Defaults to INBOX, which is the default inbox.

        Parameters:
            mailbox (str): The name of the mailbox to select.
        '''
        ok, resp = self.imap_server.login(self.email, self.password)
        if ok != 'OK':
            raise ValueError(
                f'Unable to login to mailbox {mailbox}. Please check your credentials: {str(resp)}')

        ok, resp = self.imap_server.select(mailbox)
        if ok != 'OK':
            raise ValueError(
                f'Unable to select mailbox {mailbox}. Please check the mailbox name: {str(resp)}')

    def logout(self):
        '''Shuts down the connection to the IMAP server.'''
        ok, resp = self.imap_server.logout()
        if ok != 'BYE':
            raise ValueError(
                f'Unable to logout of IMAP client: {str(resp)}')

    def search_email(self, options: EmailSearchOptions) -> pd.DataFrame:
        '''Searches emails based on the given options and returns a DataFrame.

        Parameters:
            options (EmailSearchOptions): Options to use when searching using IMAP.

        Returns:
            df (pd.DataFrame): A dataframe of emails resulting from the search.
        '''
        self.select_mailbox(options.mailbox)

        query_parts = []
        if options.subject is not None:
            query_parts.append(f'(SUBJECT "{options.subject}")')

        if options.to_email is not None:
            query_parts.append(f'(TO "{options.to_email}")')

        if options.from_email is not None:
            query_parts.append(f'(FROM "{options.from_email}")')

        if options.since_date is not None:
            since_date_str = options.since_date.strftime('%d-%b-%Y')
        else:
            since_date = datetime.today() - timedelta(days=EmailClient._DEFAULT_SINCE_DAYS)
            since_date_str = since_date.strftime('%d-%b-%Y')
        query_parts.append(f'(SINCE "{since_date_str}")')

        if options.until_date is not None:
            until_date_str = options.until_date.strftime('%d-%b-%Y')
            query_parts.append(f'(BEFORE "{until_date_str}")')

        if options.since_email_id is not None:
            query_parts.append(f'(UID {options.since_email_id}:*)')

        query = ' '.join(query_parts)
        ret = []
        _, items = self.imap_server.uid('search', None, query)
        items = items[0].split()
        for emailid in items:
            _, data = self.imap_server.uid('fetch', emailid, '(RFC822)')
            email_message = email.message_from_bytes(data[0][1])

            email_line = {}
            email_line['id'] = emailid
            email_line['to'] = email_message.get('To')
            email_line['from'] = email_message.get('From')
            email_line['subject'] = email_message.get('Subject')
            email_line['date'] = email_message.get('Date')

            plain_payload = None
            html_payload = None
            content_type = 'html'
            for part in email_message.walk():
                subtype = part.get_content_subtype()
                if subtype == 'plain':
                    # Prioritize plain text payloads when present.
                    plain_payload = part.get_payload(decode=True)
                    content_type = 'plain'
                    break
                if subtype == 'html':
                    html_payload = part.get_payload(decode=True)
            body = plain_payload or html_payload
            if body is None:
                # Very rarely messages won't have plain text or html payloads.
                continue
            email_line['body'] = plain_payload or html_payload
            email_line['body_content_type'] = content_type
            ret.append(email_line)

        return pd.DataFrame(ret)
