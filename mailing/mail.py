import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import os

from io import StringIO
from html.parser import HTMLParser


logger = logging.getLogger()

# Check if logging is already configured
if not logger:
    logging.basicConfig(
        level=logging.INFO
    )

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


class mail:
    def __init__(
        self,
        smtp_server="",   # Str: Server hostname or IP address
        TLS=False,        # Bool: Use Starttls or not
        login_pass="",    # Str: to use AUTH PLAIN, set password
        login_user="",    # Str: to use AUTH PLAIN, set username
        from_address="",  # Str: email FROM: address
        to=[],            # [Str]: email TO: addresses
        cc=[],            # [Str]: email CC: addresses
        bcc=[],           # [Str]: email BCC: addresses
        message="",       # Str: email body in HTML format
        subject="",       # Str: email subject
        attachments=[],   # [Str]: attachments to add; List of file paths
        inline_images=[]  # [(Str,Str)]: attachments added as inline images
                          # These can be referenced in HTML
                          # List of tuples: (file path, content-id) 
                          # where content-id matches the html <img src="cid:content-id"> tag
                          # content-id's should be generated with :
                          # email.utils.make_msgid(domain="example.com")
    ):
        self.smtp_server = smtp_server
        self.TLS = TLS
        self.from_address = from_address
        self.to = to
        self.cc = cc
        self.bcc = bcc
        self.message = message
        self.subject = subject
        self.attachments = attachments
        self.inline_images = inline_images
        if login_pass or login_user:
            try:
                assert login_pass
                assert login_user
            except AssertionError as e:
                logger.error("When login_pass is defined, \
                             login_user must be defined as well; \
                             and vice versa")
                raise e
        self.login_user = login_user
        self.login_pass = login_pass

    def _strip_tags(self, html):
        s = MLStripper()
        s.feed(html)
        return s.get_data()

    def send(self):
        # Multipart/related is necessary for inline images
        msg = MIMEMultipart("related")
        msg["Subject"] = self.subject
        msg["From"] = self.from_address
        msg["To"] = ",".join(self.to)
        msg["Cc"] = ",".join(self.cc)

        rcpt = self.to + self.cc + self.bcc
        plain = self._strip_tags(self.message)
        html = self.message

        # Add a multipart/alternative container  with plain text and html
        # inside the multipart/related container
        msgbody = MIMEMultipart("alternative")
        msgbody.attach(MIMEText(plain, "plain"))
        msgbody.attach(MIMEText(html, "html"))
        msg.attach(msgbody)

        # Attachments are added to the root container: multipart/related
        # Content-Disposition = inline
        if self.inline_images:
            for image_tuple in self.inline_images:
                try:
                    imagefile,content_id = image_tuple
                    with open(imagefile, "rb") as fp:
                        image = MIMEImage(fp.read())
                        image.add_header("Content-ID",content_id)
                        image.add_header("Content-Disposition", "inline")
                    msg.attach(image)
                except Exception as e:
                    logger.error(f"Could not attach inline image.")
                    raise e

        # Content-Disposition = attachment
        if self.attachments:
            for file in self.attachments:
                try:
                    with open(file, "rb") as fp:
                        attachment = MIMEBase("application", "octet-stream")
                        attachment.set_payload(fp.read())
                    encoders.encode_base64(attachment)
                    attachment.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=os.path.basename(file)
                    )
                    msg.attach(attachment)
                except Exception as e:
                    logger.error(f"Could not attach attachments.")
                    raise e

        server = smtplib.SMTP(self.smtp_server)
        server.ehlo()
        if self.TLS:
            server.starttls()
            server.ehlo()
        if self.login_pass:
            server.login(self.login_user,self.login_pass)
        server.sendmail(self.from_address, rcpt, msg.as_string())
        server.close()
