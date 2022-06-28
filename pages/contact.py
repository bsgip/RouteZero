import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, dcc, dash_table, callback
import ssl, smtplib

dash.register_page(__name__)

email_input = dbc.Row([
        dbc.Label("Email"
                , html_for="example-email-row"
                , width=2),
        dbc.Col(dbc.Input(
                type="email"
                , id="example-email-row"
                , placeholder="Enter email"
            ),width=10,
        )],className="mb-3"
)

user_input = dbc.Row([
        dbc.Label("Name", html_for="example-name-row", width=2),
        dbc.Col(
            dbc.Input(
                type="text"
                , id="example-name-row"
                , placeholder="Enter name"
                , maxLength = 80
            ),width=10
        )], className="mb-3"
)

message = dbc.Row([
        dbc.Label("Message"
         , html_for="example-message-row", width=2)
        ,dbc.Col(
            dbc.Textarea(id = "example-message-row"
                , className="mb-3"
                , placeholder="Enter message"
                , required = True)
            , width=10)
        ], className="mb-3")


def contact_form():
    markdown = ''' ### Send a message if you have a question or Feedback'''
    form = html.Div([dbc.Container([
        dcc.Markdown(markdown)
        , html.Br()
        , dbc.Card(
            dbc.CardBody([
                dbc.Form([email_input
                             , user_input
                             , message])
                , html.Div(id='div-button', children=[
                    dbc.Button('Submit'
                               , color='primary'
                               , id='button-submit'
                               , n_clicks=0)
                ])  # end div
            ])  # end cardbody
        )  # end card
        , html.Br()
        , html.Br()
    ])
    ])

    return form


@callback(Output('div-button', 'children'),
              Input("button-submit", 'n_clicks')
    , Input("example-email-row", 'value')
    , Input("example-name-row", 'value')
    , Input("example-message-row", 'value')
              )
def submit_message(n, email, name, message):
    if n > 0:
        smtp_server = "smtp.gmail.com"  # for Gmail
        port = 587  # For starttls

        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg["Subject"] = "RouteZero contact form"
        test = """
        Name: {name}
        email: {email}
        Contents: {contents}
        """.format(name=name, email=email, contents=message)


        from email.mime.text import MIMEText
        body_text = MIMEText(test, 'plain')
        msg.attach(body_text)  # attaching the text body into ms

        sender_email = "routezero.ebus@gmail.com"  # email address used to generate password
        receiver_email = ["johannes.hendriks@anu.edu.au"]  # a list of recipients
        password = "ehwsemqyudyitpzs"  # the 16 code generated

        context = ssl.create_default_context()

        try:
            server = smtplib.SMTP(smtp_server, port)
            server.ehlo()  # check connection
            server.starttls(context=context)  # Secure the connection
            server.ehlo()  # check connection
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            # Print any error messages
            print(e)
            pass
        finally:
            server.quit()

        return [html.P("Message Sent")]
    else:
        return [dbc.Button('Submit', color='primary', id='button-submit', n_clicks=0)]


layout =dbc.Container([
    # dbc.Row([
    #    dbc.Col("Please contact us with any questions or feedback via email.")
    # ]),
    # dbc.Row([
    #     dbc.Col("Email: Johannes.Hendriks@anu.edu.au")
    # ])
    contact_form()
], style={"padding":"10rem"})
