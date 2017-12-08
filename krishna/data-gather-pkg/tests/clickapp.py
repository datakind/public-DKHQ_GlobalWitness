
import click

@click.group()
@click.option('--startdate')
@click.option('--enddate')
@click.option('--directory')
@click.pass_context
def cli(ctx,
        startdate,
        enddate,
		directory):

    ctx.obj['startdate'] = startdate
    ctx.obj['enddate'] = enddate
    ctx.obj['directory'] = directory

@click.command()
@click.argument('arg1', type=str)
@click.argument('arg2', type=str)
@click.argument('arg3', type=int)
@click.pass_context
def SimpleRequest(ctx,
                  arg1,
                  arg2,
                  arg3):

    """Simple request processor"""

    ctx.obj['arg1'] = arg1
    ctx.obj['arg2'] = arg2
    ctx.obj['arg3'] = arg3

cli.add_command(SimpleRequest)

if __name__ == '__main__':
    cli(obj={})
	
