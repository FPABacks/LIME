#!/bin/env python3
import logging
import mail
import jinja2
import os
import argparse
from email.utils import make_msgid

### Constants
FRM = "lime.equation@kuleuven.be"
SRV = "smtps.kuleuven.be"
#BCC = "maarten.dirickx@kuleuven.be"
DEFAULT_TO = "system@ster.kuleuven.be"
DOMAIN = "kuleuven.be"

### Argument parsing
parser = argparse.ArgumentParser(description="Send out mail")
parser.add_argument("--to", required=True, help="Recipient email")
parser.add_argument("--subject", required=True, help="Email subject")
parser.add_argument("--body", required=True, help="Email body")
parser.add_argument("--attachment", help="Optional attachment file")
args = parser.parse_args()

### Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Set recipient
to_mail = args.to if args.to else DEFAULT_TO

# Prepare email template
tl = jinja2.FileSystemLoader(searchpath="./")
tenv = jinja2.Environment(loader=tl, autoescape=True)
template = tenv.from_string(args.body)
message = template.render()

# Prepare attachments
attachments = [args.attachment] if args.attachment and os.path.exists(args.attachment) else []

# Create the email object
e_mail = mail.mail(
    smtp_server=SRV,
    from_address=FRM,
    to=[to_mail],
    cc=[],
    TLS=True,
    message=message,
    subject=args.subject,
    attachments=attachments,
    inline_images=[],
)

logger.debug(f"Sending email from: {FRM}, to: {to_mail}, subject: {args.subject}, attachment: {attachments}")
e_mail.send()
logger.info("Email sent successfully.")