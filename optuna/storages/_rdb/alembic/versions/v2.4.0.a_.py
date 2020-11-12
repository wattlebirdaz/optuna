"""empty message

Revision ID: v2.4.0.a
Revises: v1.3.0.a
Create Date: 2020-11-12 14:13:52.206009

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'v2.4.0.a'
down_revision = 'v1.3.0.a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('study_direction',
    sa.Column('study_direction_id', sa.Integer(), nullable=False),
    sa.Column('direction', sa.Enum('NOT_SET', 'MINIMIZE', 'MAXIMIZE', name='studydirection'), nullable=False),
    sa.Column('study_id', sa.Integer(), nullable=True),
    sa.Column('objective_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['study_id'], ['studies.study_id'], ),
    sa.PrimaryKeyConstraint('study_direction_id'),
    sa.UniqueConstraint('study_id', 'objective_id')
    )
    op.create_table('trial_intermediate_values',
    sa.Column('trial_intermediate_values_id', sa.Integer(), nullable=False),
    sa.Column('trial_id', sa.Integer(), nullable=True),
    sa.Column('step', sa.Integer(), nullable=True),
    sa.Column('value', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['trial_id'], ['trials.trial_id'], ),
    sa.PrimaryKeyConstraint('trial_intermediate_values_id'),
    sa.UniqueConstraint('trial_id', 'step')
    )
    op.drop_column('studies', 'direction')
    op.add_column('trial_values', sa.Column('objective_id', sa.Integer(), nullable=True))
    op.create_unique_constraint(None, 'trial_values', ['trial_id', 'objective_id'])
    op.drop_column('trial_values', 'step')
    op.drop_column('trials', 'value')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('trials', sa.Column('value', sa.FLOAT(), nullable=True))
    op.add_column('trial_values', sa.Column('step', sa.INTEGER(), nullable=True))
    op.drop_constraint(None, 'trial_values', type_='unique')
    op.drop_column('trial_values', 'objective_id')
    op.add_column('studies', sa.Column('direction', sa.VARCHAR(length=8), nullable=False))
    op.drop_table('trial_intermediate_values')
    op.drop_table('study_direction')
    # ### end Alembic commands ###
