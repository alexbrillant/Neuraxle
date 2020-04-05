from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, NonTransformableMixin
from neuraxle.hyperparams.distributions import RandInt, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace, HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import StepClonerForEachDataInput
from testing.test_pipeline import SomeStep

SOME_STEP_HP_KEY = 'somestep_hyperparam'
RAND_INT_SOME_STEP = RandInt(-10, 0)
RAND_INT_STEP_CLONER = RandInt(0, 10)

META_STEP_HP = 'metastep_hyperparam'
SOME_STEP_HP = "SomeStep__somestep_hyperparam"
META_STEP_HP_VALUE = 1
SOME_STEP_HP_VALUE = 2

HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean()
})

HYPE_SAMPLE = HyperparameterSamples({
    "a__test": True
})


class SomeMetaStepMixin(NonTransformableMixin, NonFittableMixin, MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep):
        BaseStep.__init__(self)
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped)


class SomeStepInverseTransform(SomeStep):
    def fit_transform(self, data_inputs, expected_outputs=None):
        return self, 'fit_transform'

    def inverse_transform(self, processed_outputs):
        return 'inverse_transform'


def test_step_cloner_should_get_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    hyperparams = p.get_hyperparams()

    assert hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert hyperparams[SOME_STEP_HP] == SOME_STEP_HP_VALUE


def test_step_cloner_should_set_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_update_hyperparams_should_update_step_cloner_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE + 1,
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE + 1
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_update_hyperparams_should_update_wrapped_step_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        SOME_STEP_HP: SOME_STEP_HP_VALUE + 1,
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE + 1


def test_step_cloner_update_hyperparams_space_should_update_step_cloner_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    update_meta_step_hp_space = RandInt(0, 40)
    p.update_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: update_meta_step_hp_space
    }))

    assert isinstance(p.hyperparams_space, HyperparameterSpace)
    assert p.hyperparams_space[META_STEP_HP] == update_meta_step_hp_space
    assert p.wrapped.get_hyperparams_space()[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_step_cloner_update_hyperparams_space_should_update_wrapped_step_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    updated_some_step_hp_space = RandInt(0, 400)
    p.update_hyperparams_space(HyperparameterSpace({
        SOME_STEP_HP: updated_some_step_hp_space
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert p.wrapped.get_hyperparams_space()[SOME_STEP_HP_KEY] == updated_some_step_hp_space


def test_step_cloner_should_set_steps_hyperparams():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert isinstance(p.get_step().hyperparams, HyperparameterSamples)
    assert p.get_step().get_hyperparams()[SOME_STEP_HP_KEY] == SOME_STEP_HP_VALUE


def test_step_cloner_should_set_steps_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert isinstance(p.get_step().hyperparams_space, HyperparameterSpace)
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_step_cloner_should_set_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())

    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert isinstance(p.hyperparams_space, HyperparameterSpace)
    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_STEP_CLONER
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_step_cloner_should_get_hyperparams_space():
    p = StepClonerForEachDataInput(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_STEP_CLONER,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space[META_STEP_HP] == RAND_INT_STEP_CLONER
    assert hyperparams_space[SOME_STEP_HP] == RAND_INT_SOME_STEP


RAND_INT_META_STEP = RandInt(0, 10)


def test_pipeline_should_set_hyperparams():
    p = Pipeline([
        SomeStep().set_name('step_1'),
        SomeStep().set_name('step_2')
    ])

    p.set_hyperparams(HyperparameterSamples({
        'hp': 1,
        'step_1__hp': 2,
        'step_2__hp': 3
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams['hp'] == 1
    assert p[0].hyperparams['hp'] == 2
    assert p[1].hyperparams['hp'] == 3


def test_pipeline_should_set_hyperparams_space():
    p = Pipeline([
        SomeStep().set_name('step_1'),
        SomeStep().set_name('step_2')
    ])

    p.set_hyperparams_space(HyperparameterSpace({
        'hp': RandInt(1, 2),
        'step_1__hp': RandInt(2, 3),
        'step_2__hp': RandInt(3, 4)
    }))

    assert isinstance(p.hyperparams_space, HyperparameterSpace)

    assert p.hyperparams_space['hp'].min_included == 1
    assert p.hyperparams_space['hp'].max_included == 2

    assert p[0].hyperparams_space['hp'].min_included == 2
    assert p[0].hyperparams_space['hp'].max_included == 3

    assert p[1].hyperparams_space['hp'].min_included == 3
    assert p[1].hyperparams_space['hp'].max_included == 4


def test_pipeline_should_update_hyperparams():
    p = Pipeline([
        SomeStep().set_name('step_1'),
        SomeStep().set_name('step_2')
    ])

    p.set_hyperparams(HyperparameterSamples({
        'hp': 1,
        'step_1__hp': 2,
        'step_2__hp': 3
    }))

    p.update_hyperparams(HyperparameterSamples({
        'hp': 4,
        'step_2__hp': 6
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams['hp'] == 4
    assert p[0].hyperparams['hp'] == 2
    assert p[1].hyperparams['hp'] == 6


def test_pipeline_should_update_hyperparams_space():
    p = Pipeline([
        SomeStep().set_name('step_1'),
        SomeStep().set_name('step_2')
    ])

    p.set_hyperparams_space(HyperparameterSpace({
        'hp': RandInt(1, 2),
        'step_1__hp': RandInt(2, 3),
        'step_2__hp': RandInt(3, 4)
    }))
    p.update_hyperparams_space(HyperparameterSpace({
        'hp': RandInt(4, 6),
        'step_2__hp': RandInt(6, 8)
    }))

    assert isinstance(p.hyperparams_space, HyperparameterSpace)

    assert p.hyperparams_space['hp'].min_included == 4
    assert p.hyperparams_space['hp'].max_included == 6

    assert p[0].hyperparams_space['hp'].min_included == 2
    assert p[0].hyperparams_space['hp'].max_included == 3

    assert p[1].hyperparams_space['hp'].min_included == 6
    assert p[1].hyperparams_space['hp'].max_included == 8


def test_meta_step_mixin_should_get_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    hyperparams = p.get_hyperparams()

    assert hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert hyperparams[SOME_STEP_HP] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_should_set_hyperparams():
    p = SomeMetaStepMixin(SomeStep())

    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    assert isinstance(p.hyperparams, HyperparameterSamples)
    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_update_hyperparams_should_update_meta_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE + 1
    }))

    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE + 1
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE


def test_meta_step_mixin_update_hyperparams_should_update_wrapped_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams(HyperparameterSamples({
        META_STEP_HP: META_STEP_HP_VALUE,
        SOME_STEP_HP: SOME_STEP_HP_VALUE
    }))

    p.update_hyperparams(HyperparameterSamples({
        SOME_STEP_HP: SOME_STEP_HP_VALUE + 1
    }))

    assert p.hyperparams[META_STEP_HP] == META_STEP_HP_VALUE
    assert p.get_step().get_hyperparams()['somestep_hyperparam'] == SOME_STEP_HP_VALUE + 1


def test_meta_step_mixin_update_hyperparams_space_should_update_meta_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    update_meta_step_hp_space = RandInt(0, 100)
    p.update_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: update_meta_step_hp_space
    }))

    assert p.hyperparams_space[META_STEP_HP] == update_meta_step_hp_space
    assert p.wrapped.get_hyperparams_space()['somestep_hyperparam'] == RAND_INT_SOME_STEP


def test_meta_step_mixin_update_hyperparams_space_should_update_wrapped_step_hyperparams():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    updated_some_step_hp_space = RandInt(0, 100)
    p.update_hyperparams_space(HyperparameterSpace({
        SOME_STEP_HP: updated_some_step_hp_space
    }))

    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert p.wrapped.get_hyperparams_space()['somestep_hyperparam'] == updated_some_step_hp_space


def test_meta_step_mixin_should_set_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    assert p.hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert p.get_step().hyperparams_space[SOME_STEP_HP_KEY] == RAND_INT_SOME_STEP


def test_meta_step_mixin_should_get_hyperparams_space():
    p = SomeMetaStepMixin(SomeStep())
    p.set_hyperparams_space(HyperparameterSpace({
        META_STEP_HP: RAND_INT_META_STEP,
        SOME_STEP_HP: RAND_INT_SOME_STEP
    }))

    hyperparams_space = p.get_hyperparams_space()

    assert hyperparams_space[META_STEP_HP] == RAND_INT_META_STEP
    assert hyperparams_space[SOME_STEP_HP] == RAND_INT_SOME_STEP


def test_get_set_params_base_step():
    s = SomeStep()

    s.set_params(learning_rate=0.1)
    hyperparams = s.get_params()

    assert hyperparams == {"learning_rate": 0.1}
